import os

from ultralytics.data import build_yolo_dataset
from ultralytics.utils import callbacks as callbacks_ult
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics import YOLO
from ultralytics.utils import IterableSimpleNamespace

from pathlib import Path
import torch
from typing import List, Dict, Union
import numpy as np

import package_file
test_pr,cfg,all_clss,wanted_cls_dic,possible_float_like_nan_types=package_file.set_global_params()






# ----------------------------------------------------data processing---------------------------------------------------

@package_file.tensorleap_preprocess()
def preprocess_func_leap() -> List[package_file.PreprocessResponse]:
    dataset_types = [package_file.DataStateType.training, package_file.DataStateType.validation]
    phases = ['train', 'val']
    responses = []
    if cfg.tensorleap_use_test:
        phases.append('test')
        dataset_types.append(package_file.DataStateType.test)
    if cfg.tensorleap_use_unlabeled:
        phases.append('unlabeled')
        dataset_types.append(package_file.DataStateType.unlabeled)
    for phase, dataset_type in zip(phases, dataset_types):
        data_loader, n_samples = package_file.create_data_with_ult(cfg, yolo_data, phase=phase)
        responses.append(
            package_file.PreprocessResponse(sample_ids=list(range(n_samples)),
                               data={'dataloader':data_loader},
                               sample_id_type=int,
                               state=dataset_type))
    return responses


# ------------------------------------------input and gt----------------------------------------------------------------

@package_file.tensorleap_input_encoder('image',channel_dim=1)
def input_encoder(idx: int, preprocess: package_file.PreprocessResponse) -> np.ndarray:
    imgs, _, _,_=package_file.pre_process_dataloader(preprocess, idx, predictor)
    return imgs.astype('float32')


@package_file.tensorleap_gt_encoder('classes')
def gt_encoder(idx: int, preprocessing: package_file.PreprocessResponse) -> np.ndarray:
    _, clss, bboxes, _ =package_file.pre_process_dataloader(preprocessing, idx,predictor)
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

@package_file.tensorleap_metadata('metadata_sample_index')
def metadata_sample_index(idx: int, preprocess: package_file.PreprocessResponse) -> int:
    return idx


@package_file.tensorleap_metadata("image info a", metadata_type = possible_float_like_nan_types)
def metadata_per_img(idx: int, data: package_file.PreprocessResponse) -> Dict[str, Union[str, int, float]]:
    nan_default_value = None
    gt_data = gt_encoder(idx, data)
    cls_gt = np.expand_dims(gt_data[:, 4], axis=1)
    bbox_gt = gt_data[:, :4]
    clss_info = np.unique(cls_gt, return_counts=True)
    count_dict = package_file.update_dict_count_cls(all_clss, clss_info,nan_default_value)
    areas, aspect_ratios = package_file.bbox_area_and_aspect_ratio(bbox_gt, data.data['dataloader'][idx]['resized_shape'])
    occlusion_matrix, areas_in_pixels, union_in_pixels = package_file.calculate_iou_all_pairs(bbox_gt, data.data['dataloader'][idx][
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

@package_file.tensorleap_custom_loss("total_loss")
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
@package_file.tensorleap_custom_visualizer("bb_gt_decoder", package_file.LeapDataType.ImageWithBBox)
def gt_bb_decoder(image: np.ndarray, bb_gt: np.ndarray) -> package_file.LeapImageWithBBox:
    bbox = [package_file.BoundingBox(x=bbx[0], y=bbx[1], width=bbx[2], height=bbx[3], confidence=1, label=all_clss.get(int(bbx[4]) if not np.isnan(bbx[4]) else -1, 'Unknown Class')) for bbx in bb_gt.squeeze(0)]
    image = package_file.rescale_min_max(image.squeeze(0))
    return package_file.LeapImageWithBBox(data=(image.transpose(1,2,0)), bounding_boxes=bbox)


@package_file.tensorleap_custom_visualizer('image_visualizer', package_file.LeapDataType.Image)
def image_visualizer(image: np.ndarray) -> package_file.LeapImage:
    image = package_file.rescale_min_max(image.squeeze(0))
    return package_file.LeapImage((image.transpose(1,2,0)), compress=False)


@package_file.tensorleap_custom_visualizer("bb_decoder", package_file.LeapDataType.ImageWithBBox)
def bb_decoder(image: np.ndarray, predictions: np.ndarray) -> package_file.LeapImageWithBBox:
    image=image.squeeze(0)
    y_pred = predictor.postprocess(torch.from_numpy(predictions))
    _, cls_temp, bbx_temp, conf_temp = package_file.output_to_target(y_pred, max_det=predictor.args.max_det)
    t_pred = np.concatenate([bbx_temp, np.expand_dims(conf_temp, 1), np.expand_dims(cls_temp, 1)], axis=1)
    post_proc_pred = t_pred[t_pred[:, 4] >  (getattr(cfg, "conf", 0.25) or 0.25)]
    post_proc_pred[:, :4:2] /= image.shape[1]
    post_proc_pred[:, 1:4:2] /= image.shape[2]
    bbox = [package_file.BoundingBox(x=bbx[0], y=bbx[1], width=bbx[2], height=bbx[3], confidence=bbx[4], label=all_clss.get(int(bbx[5]),'Unknown Class')) for bbx in post_proc_pred]
    image = package_file.rescale_min_max(image)
    return package_file.LeapImageWithBBox(data=(image.transpose(1,2,0)), bounding_boxes=bbox)


#Greedy one2one iou
@package_file.tensorleap_custom_metric("ious", direction=package_file.MetricDirection.Upward)
def ious(y_pred: np.ndarray,preprocess: package_file.SamplePreprocessResponse):
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
    iou_mat = package_file.box_iou(boxes_gt, predn[:, :4]).numpy()
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



@package_file.tensorleap_custom_metric("cost", direction=package_file.MetricDirection.Downward)
def cost(pred80,pred40,pred20,gt):
    gt=np.squeeze(gt,axis=0)
    d={}
    d["bboxes"] = torch.from_numpy(gt[...,:4])
    d["cls"] = torch.from_numpy(gt[...,4])
    d["batch_idx"] = torch.zeros_like(d['cls'])
    y_pred_torch = [torch.from_numpy(s) for s in [pred80,pred40,pred20]]
    _,loss_parts= criterion(y_pred_torch, d)
    return {"box":loss_parts[0].unsqueeze(0).numpy(),"cls":loss_parts[1].unsqueeze(0).numpy(),"dfl":loss_parts[2].unsqueeze(0).numpy()}


@package_file.tensorleap_custom_metric('Confusion Matrix')
def confusion_matrix_metric(y_pred: np.ndarray, preprocess: package_file.SamplePreprocessResponse):
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
        ious = package_file.box_iou(bbox, predn[:, :4]).numpy().T
        prediction_detected = np.any((ious > threshold), axis=1)
        max_iou_ind = np.argmax(ious, axis=1)
        for i, prediction in enumerate(prediction_detected):
            gt_idx = int(batch['cls'][max_iou_ind[i]])
            class_name = all_clss.get(gt_idx)
            gt_label = f"{class_name}"
            confidence = predn[i, 4]
            if prediction:  # TP
                confusion_matrix_elements.append(package_file.ConfusionMatrixElement(
                    str(gt_label),
                    package_file.ConfusionMatrixValue.Positive,
                    float(confidence)
                ))
            else:  # FP
                class_name = all_clss.get(int(predn[i,5]))
                pred_label = f"{class_name}"
                confusion_matrix_elements.append(package_file.ConfusionMatrixElement(
                    str(pred_label),
                    package_file.ConfusionMatrixValue.Negative,
                    float(confidence)
                ))
    else:  # No prediction
        ious = np.zeros((1, cls.shape[0]))
    gts_detected = np.any((ious > threshold), axis=0)
    for k, gt_detection in enumerate(gts_detected):
        label_idx = cls[k]
        if not gt_detection : # FN
            class_name = all_clss.get(int(label_idx))
            confusion_matrix_elements.append(package_file.ConfusionMatrixElement(
                f"{class_name}",
                package_file.ConfusionMatrixValue.Positive,
                float(0)
            ))
    if all(~ gts_detected):
        confusion_matrix_elements.append(package_file.ConfusionMatrixElement(
            "background",
            package_file.ConfusionMatrixValue.Positive,
            float(0)
        ))
    return [confusion_matrix_elements]


# ---------------------------------------------------------ult dependent code------------------------------------------------------

def get_yolo_data(cfg):
    from ultralytics.data.utils import check_det_dataset
    return check_det_dataset(cfg.data, autodownload=True)


def get_criterion(cfg):
    model_path=Path(cfg.model)
    if not model_path.is_absolute():
        model_path = (cfg.tensorleap_path / model_path).resolve()
    if not test_pr:
        assert model_path.is_relative_to(cfg.tensorleap_path), (
            f"❌ {model_path!r} is not inside tensorleap path {cfg.tensorleap_path!r}" )
    model_base = YOLO(model_path)
    criterion = model_base.init_criterion()
    criterion.hyp = IterableSimpleNamespace(**criterion.hyp)
    criterion.hyp.box = cfg.box
    criterion.hyp.cls = cfg.cls
    criterion.hyp.dfl = cfg.dfl
    return criterion

def create_data_with_ult(cfg,yolo_data, phase='val'):
    n_samples = len(os.listdir(yolo_data[phase]))
    dataset = build_yolo_dataset(cfg, yolo_data[phase],n_samples , yolo_data, mode='val', stride=32)
    return dataset, n_samples

def get_predictor_obj(cfg,yolo_data):
    callbacks = callbacks_ult.get_default_callbacks()
    predictor = DetectionValidator(args=cfg, _callbacks=callbacks)
    predictor.data = yolo_data
    predictor.end2end = False
    return predictor

yolo_data = get_yolo_data(cfg)
predictor = get_predictor_obj(cfg,yolo_data)
criterion = get_criterion(cfg)
# ---------------------------------------------------------main------------------------------------------------------


package_file.leap_binder.add_prediction(name='object detection', labels=["x", "y", "w", "h"] + [cl for cl in all_clss.values()], channel_dim=1)
package_file.leap_binder.add_prediction(name='concatenate_20', labels=[str(i) for i in range(20)], channel_dim=-1)
package_file.leap_binder.add_prediction(name='concatenate_40', labels=[str(i) for i in range(40)], channel_dim=-1)
package_file.leap_binder.add_prediction(name='concatenate_80', labels=[str(i) for i in range(80)], channel_dim=-1)

# if __name__ == '__main__':
#     leap_binder.check()

