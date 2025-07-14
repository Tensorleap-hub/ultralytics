import os

import torch
from typing import List, Dict, Union
import numpy as np
import tensorflow as tf
import onnxruntime as ort
from code_loader.helpers import visualize

from ult_utils import get_global_params
from ult_utils import create_data_with_ult, pre_process_dataloader, \
    update_dict_count_cls, bbox_area_and_aspect_ratio, calculate_iou_all_pairs, validate_supported_models, extract_mapping, onnx_exporter

from code_loader import leap_binder
from code_loader.contract.datasetclasses import PreprocessResponse, ConfusionMatrixElement
from code_loader.contract.enums import LeapDataType, MetricDirection, ConfusionMatrixValue
from code_loader.visualizers.default_visualizers import LeapImage
from code_loader.inner_leap_binder.leapbinder_decorators import (tensorleap_preprocess, tensorleap_gt_encoder,
                                                                 tensorleap_input_encoder, tensorleap_metadata,
                                                                 tensorleap_custom_visualizer)
from code_loader.contract.responsedataclasses import BoundingBox
from code_loader.contract.visualizer_classes import LeapImageWithBBox
from code_loader.utils import rescale_min_max
from code_loader.contract.datasetclasses import SamplePreprocessResponse
from code_loader.contract.enums import DataStateType
from code_loader.inner_leap_binder.leapbinder_decorators import tensorleap_custom_loss, tensorleap_custom_metric


from ultralytics.utils.plotting import output_to_target #doable
from ultralytics.utils.metrics import box_iou #doable



dir_path, cfg, yolo_data, dataset_yaml, criterion, all_clss, cls_mapping, wanted_cls_dic, predictor, possible_float_like_nan_types=get_global_params()

# ----------------------------------------------------data processing---------------------------------------------------

@tensorleap_preprocess()
def preprocess_func_leap() -> List[PreprocessResponse]:
    dataset_types = [DataStateType.training, DataStateType.validation]
    phases = ['train', 'val']
    responses = []
    if cfg.tensorleap_use_test:
        phases.append('test')
        dataset_types.append(DataStateType.test)
    if cfg.tensorleap_use_unlabeled:
        phases.append('unlabeled')
        dataset_types.append(DataStateType.unlabeled)
    for phase, dataset_type in zip(phases, dataset_types):
        data_loader, n_samples = create_data_with_ult(cfg, yolo_data, phase=phase)
        responses.append(
            PreprocessResponse(sample_ids=list(range(n_samples)),
                               data={'dataloader':data_loader},
                               sample_id_type=int,
                               state=dataset_type))
    return responses


# ------------------------------------------input and gt----------------------------------------------------------------

@tensorleap_input_encoder('image',channel_dim=1)
def input_encoder(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
    imgs, _, _,_=pre_process_dataloader(preprocess, idx, predictor)
    return imgs.astype('float32')


@tensorleap_gt_encoder('classes')
def gt_encoder(idx: int, preprocessing: PreprocessResponse) -> np.ndarray:
    _, clss, bboxes, _ =pre_process_dataloader(preprocessing, idx,predictor)
    if clss.shape[0]==0 and  bboxes.shape[0]==0:
        return np.full((1, 5), np.nan,dtype=np.float32)
    elif clss.shape[0]==0:
        temp_array=np.full((bboxes.shape[0], 5), np.nan,dtype=np.float32)
        temp_array[:,:4]=bboxes
        return temp_array
    elif bboxes.shape[0]==0:
        temp_array = np.full((clss.shape[0], 5), np.nan,dtype=np.float32)
        temp_array[:, 4] = clss
        return temp_array
    return np.concatenate([bboxes,clss],axis=1)

# ----------------------------------------------------------metadata----------------------------------------------------

@tensorleap_metadata('metadata_sample_index')
def metadata_sample_index(idx: int, preprocess: PreprocessResponse) -> int:
    return idx


@tensorleap_metadata("image info a", metadata_type = possible_float_like_nan_types)
def metadata_per_img(idx: int, data: PreprocessResponse) -> Dict[str, Union[str, int, float]]:
    nan_default_value = None
    gt_data = gt_encoder(idx, data)
    cls_gt = np.expand_dims(gt_data[:, 4], axis=1)
    bbox_gt = gt_data[:, :4]
    clss_info = np.unique(cls_gt, return_counts=True)
    count_dict = update_dict_count_cls(all_clss, clss_info,nan_default_value)
    areas, aspect_ratios = bbox_area_and_aspect_ratio(bbox_gt, data.data['dataloader'][idx]['resized_shape'])
    occlusion_matrix, areas_in_pixels, union_in_pixels = calculate_iou_all_pairs(bbox_gt, data.data['dataloader'][idx][
        'resized_shape'])
    no_nans_values = ~np.isnan(clss_info[0]).any()
    d = {
        "image path": data.data['dataloader'].im_files[idx],
        "idx": idx,
        "# unique classes": len(clss_info[0]) if no_nans_values else nan_default_value,
        "# of objects": int(clss_info[1].sum()) if no_nans_values else nan_default_value,
        "mean bbox area": float(areas.mean()) if no_nans_values else nan_default_value,
        "var bbox area": float(areas.var()) if no_nans_values else nan_default_value,
        "median bbox area": float(np.median(areas)) if no_nans_values else nan_default_value,
        "max bbox area": float(np.max(areas)) if no_nans_values else nan_default_value,
        "min bbox area": float(np.min(areas)) if no_nans_values else nan_default_value,
        "bbox overlap": float(
            occlusion_matrix.sum() / areas_in_pixels.sum()) if no_nans_values else nan_default_value,
        "max bbox overlap": float(
            (occlusion_matrix.sum(axis=1) / areas_in_pixels).max()) if no_nans_values else nan_default_value,
    }
    d.update(**count_dict)
    return d



# ----------------------------------------------------------loss--------------------------------------------------------

@tensorleap_custom_loss("total_loss")
def loss(pred80,pred40,pred20,gt,demo_pred):
    gt=np.squeeze(gt,axis=0)
    d={}
    d["bboxes"] = torch.from_numpy(gt[...,:4])
    d["cls"] = torch.from_numpy(gt[...,4])
    d["batch_idx"] = torch.zeros_like(d['cls'])
    y_pred_torch = [torch.from_numpy(s) for s in [pred80,pred40,pred20]]
    all_loss,_= criterion(y_pred_torch, d)
    return all_loss.unsqueeze(0).numpy()


# ------------------------------------------------------visualizers-----------------------------------------------------
@tensorleap_custom_visualizer("bb_gt_decoder", LeapDataType.ImageWithBBox)
def gt_bb_decoder(image: np.ndarray, bb_gt: np.ndarray) -> LeapImageWithBBox:
    bbox = [BoundingBox(x=bbx[0], y=bbx[1], width=bbx[2], height=bbx[3], confidence=1, label=all_clss.get(int(bbx[4]) if not np.isnan(bbx[4]) else -1, 'Unknown Class')) for bbx in bb_gt.squeeze(0)]
    image = rescale_min_max(image.squeeze(0))
    return LeapImageWithBBox(data=(image.transpose(1,2,0)), bounding_boxes=bbox)


@tensorleap_custom_visualizer('image_visualizer', LeapDataType.Image)
def image_visualizer(image: np.ndarray) -> LeapImage:
    image = rescale_min_max(image.squeeze(0))
    return LeapImage((image.transpose(1,2,0)), compress=False)


@tensorleap_custom_visualizer("bb_decoder", LeapDataType.ImageWithBBox)
def bb_decoder(image: np.ndarray, predictions: np.ndarray) -> LeapImageWithBBox:
    image=image.squeeze(0)
    y_pred = predictor.postprocess(torch.from_numpy(predictions))
    _, cls_temp, bbx_temp, conf_temp = output_to_target(y_pred, max_det=predictor.args.max_det)
    t_pred = np.concatenate([bbx_temp, np.expand_dims(conf_temp, 1), np.expand_dims(cls_temp, 1)], axis=1)
    post_proc_pred = t_pred[t_pred[:, 4] >  (getattr(cfg, "conf", 0.25) or 0.25)]
    post_proc_pred[:, :4:2] /= image.shape[1]
    post_proc_pred[:, 1:4:2] /= image.shape[2]
    bbox = [BoundingBox(x=bbx[0], y=bbx[1], width=bbx[2], height=bbx[3], confidence=bbx[4], label=all_clss.get(int(bbx[5]),'Unknown Class')) for bbx in post_proc_pred]
    image = rescale_min_max(image)
    return LeapImageWithBBox(data=(image.transpose(1,2,0)), bounding_boxes=bbox)


#Greedy one2one iou
@tensorleap_custom_metric("ious", direction=MetricDirection.Upward)
def ious(y_pred: np.ndarray,preprocess: SamplePreprocessResponse):
    default_value =  np.ones(1) * -1 # TODO - set to NONE
    batch = preprocess.preprocess_response.data['dataloader'][int(preprocess.sample_ids)]
    batch["imgsz"]     = (batch["resized_shape"],)
    batch["ori_shape"] = (batch["ori_shape"],)
    batch["ratio_pad"] = (batch["ratio_pad"],)
    batch["img"]       = batch["img"].unsqueeze(0)
    pred = predictor.postprocess(torch.from_numpy(y_pred))[0]
    predictor.seen, predictor.args.plots, predictor.stats = 0, False, {"tp": []}
    pbatch = predictor._prepare_batch(0, batch)
    wanted_mask = np.isin(pbatch['cls'].numpy(),
                          np.array(list(wanted_cls_dic.values())))
    cls_gt, boxes_gt = pbatch.pop("cls"), pbatch.pop("bbox")
    predn   = predictor._prepare_pred(pred, pbatch)
    iou_dic = dict.fromkeys(wanted_cls_dic.keys(), default_value)
    if boxes_gt.shape[0] == 0 and predn.shape[0] == 0:
        iou_dic["mean sample iou"] = default_value
        return iou_dic
    iou_mat = box_iou(boxes_gt, predn[:, :4]).numpy()
    n_gt, n_pred = iou_mat.shape
    used_gt = np.zeros(n_gt, dtype=bool)
    assigned_iou_per_gt = np.zeros(n_gt)
    iou_per_pred = np.zeros(n_pred)
    for j in range(n_pred):
        i = np.argmax(iou_mat[:, j])
        best = iou_mat[i, j]
        if not used_gt[i]:
            iou_per_pred[j] = best
            assigned_iou_per_gt[i] = best
            used_gt[i] = True
    all_instance_ious = np.concatenate([iou_per_pred, np.zeros(np.sum(~used_gt))])
    mean_iou_sample   = np.expand_dims(all_instance_ious.mean(), axis=0)
    for c_id, c_name in wanted_cls_dic.items():
        mask_c = (cls_gt.numpy() == c_name) & wanted_mask
        if mask_c.any():
            iou_dic[c_id] = np.expand_dims(assigned_iou_per_gt[mask_c].mean(), axis=0)

    iou_dic["mean sample iou"] = mean_iou_sample
    return iou_dic



@tensorleap_custom_metric("cost", direction=MetricDirection.Downward)
def cost(pred80,pred40,pred20,gt):
    gt=np.squeeze(gt,axis=0)
    d={}
    d["bboxes"] = torch.from_numpy(gt[...,:4])
    d["cls"] = torch.from_numpy(gt[...,4])
    d["batch_idx"] = torch.zeros_like(d['cls'])
    y_pred_torch = [torch.from_numpy(s) for s in [pred80,pred40,pred20]]
    _,loss_parts= criterion(y_pred_torch, d)
    return {"box":loss_parts[0].unsqueeze(0).numpy(),"cls":loss_parts[1].unsqueeze(0).numpy(),"dfl":loss_parts[2].unsqueeze(0).numpy()}


@tensorleap_custom_metric('Confusion Matrix')
def confusion_matrix_metric(y_pred: np.ndarray, preprocess: SamplePreprocessResponse):
    threshold=cfg.iou
    confusion_matrix_elements = []
    batch=preprocess.preprocess_response.data['dataloader'][int(preprocess.sample_ids)]
    batch["imgsz"]=(batch["resized_shape"],)
    batch["ori_shape"]=(batch["ori_shape"],)
    batch["ratio_pad"]= (batch["ratio_pad"],)
    batch["img"]=batch["img"].unsqueeze(0)
    pred = predictor.postprocess(torch.from_numpy(y_pred))[0]
    predictor.seen=0
    predictor.args.plots=False
    predictor.stats={}
    predictor.stats['tp']=[]
    pbatch = predictor._prepare_batch(0, batch)
    cls, bbox = pbatch.pop("cls"), pbatch.pop("bbox")
    predn = predictor._prepare_pred(pred, pbatch)
    if len(predn)!=0:
        ious = box_iou(bbox, predn[:, :4]).numpy().T
        prediction_detected = np.any((ious > threshold), axis=1)
        max_iou_ind = np.argmax(ious, axis=1)
        for i, prediction in enumerate(prediction_detected):
            gt_idx = int(batch['cls'][max_iou_ind[i]])
            class_name = all_clss.get(gt_idx)
            gt_label = f"{class_name}"
            confidence = predn[i, 4]
            if prediction:  # TP
                confusion_matrix_elements.append(ConfusionMatrixElement(
                    str(gt_label),
                    ConfusionMatrixValue.Positive,
                    float(confidence)
                ))
            else:  # FP
                class_name = all_clss.get(int(predn[i,5]))
                pred_label = f"{class_name}"
                confusion_matrix_elements.append(ConfusionMatrixElement(
                    str(pred_label),
                    ConfusionMatrixValue.Negative,
                    float(confidence)
                ))
    else:  # No prediction
        ious = np.zeros((1, cls.shape[0]))
    gts_detected = np.any((ious > threshold), axis=0)
    for k, gt_detection in enumerate(gts_detected):
        label_idx = cls[k]
        if not gt_detection : # FN
            class_name = all_clss.get(int(label_idx))
            confusion_matrix_elements.append(ConfusionMatrixElement(
                f"{class_name}",
                ConfusionMatrixValue.Positive,
                float(0)
            ))
    if all(~ gts_detected):
        confusion_matrix_elements.append(ConfusionMatrixElement(
            "background",
            ConfusionMatrixValue.Positive,
            float(0)
        ))
    return [confusion_matrix_elements]






def run_inference(
        check_generic = True,
        plot_vis = True,
        model_path = None , # Choose None if only pt version available else, use your h5/onnx model's path.
        mapping_version = None
                   ):
        if check_generic:
            leap_binder.check()
        m_path = model_path if model_path != None else 'None_path'
        print("started custom tests")
        validate_supported_models(os.path.basename(cfg.model), m_path)
        if not os.path.exists(m_path):
            m_path = onnx_exporter(cfg)
            extract_mapping(m_path, mapping_version)
        keras_model = m_path.endswith(".h5")
        model = tf.keras.models.load_model(m_path) if keras_model else ort.InferenceSession(m_path)
        responses = preprocess_func_leap()
        for subset in responses:  # [training, validation, test ,unlabeled]
            for idx in range(4):
                s_prepro = SamplePreprocessResponse(np.array(idx), subset)

                # get input images
                image = input_encoder(idx, subset)
                concat = np.expand_dims(image, axis=0)

                # predict
                y_pred = model([concat]) if keras_model else model.run(None, {model.get_inputs()[0].name: concat})
                if not keras_model:
                    y_pred = [tf.convert_to_tensor(p) for p in y_pred]
                if subset.state != DataStateType.unlabeled:
                    # get gt
                    gt = gt_encoder(idx, subset)
                    gt_img = gt_bb_decoder(np.expand_dims(image, axis=0), np.expand_dims(gt, axis=0))

                    # custom metrics
                    total_loss = loss(y_pred[1].numpy(), y_pred[2].numpy(), y_pred[3].numpy(),
                                      np.expand_dims(gt, axis=0), y_pred[0].numpy())
                    cost_dic = cost(y_pred[1].numpy(), y_pred[2].numpy(), y_pred[3].numpy(), np.expand_dims(gt, axis=0))
                    iou = ious(y_pred[0].numpy(), s_prepro)
                    conf_mat = confusion_matrix_metric(y_pred[0].numpy(), s_prepro)

                # metadata
                meta_data = metadata_per_img(idx, subset)

                # vis
                img_vis = image_visualizer(np.expand_dims(image, axis=0))
                pred_img = bb_decoder(np.expand_dims(image, axis=0), y_pred[0].numpy())
                if plot_vis:
                    visualize(img_vis)
                    visualize(pred_img)
                    if subset.state != DataStateType.unlabeled:
                        visualize(gt_img)
        print("finish tests")


# ---------------------------------------------------------main------------------------------------------------------



leap_binder.add_prediction(name='object detection', labels=["x", "y", "w", "h"] + [cl for cl in all_clss.values()], channel_dim=1)
leap_binder.add_prediction(name='concatenate_20', labels=[str(i) for i in range(20)], channel_dim=-1)
leap_binder.add_prediction(name='concatenate_40', labels=[str(i) for i in range(40)], channel_dim=-1)
leap_binder.add_prediction(name='concatenate_80', labels=[str(i) for i in range(80)], channel_dim=-1)

if __name__ == '__main__':
    run_inference(cfg)
    leap_binder.check()

