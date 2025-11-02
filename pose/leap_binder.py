import copy

import torch
from code_loader.inner_leap_binder.leapbinder_decorators import tensorleap_custom_loss, tensorleap_custom_metric
from numpy import ndarray, dtype

from ultralytics.engine.results import Boxes
from ultralytics.tensorleap_folder.global_params import cfg, yolo_data, criterion, all_clss, \
    possible_float_like_nan_types, wanted_cls_dic, predictor, ob_yolo_data, ob_cfg, ob_all_clss
from ultralytics.tensorleap_folder.utils import create_data_with_ult, pre_process_dataloader, \
    update_dict_count_cls, bbox_area_and_aspect_ratio, calculate_iou_all_pairs, pre_process_ob_dataloader
from typing import List, Dict, Union, Any
import numpy as np
from code_loader.contract.datasetclasses import PreprocessResponse, DataStateType, SamplePreprocessResponse, \
    ConfusionMatrixElement
from code_loader.contract.enums import LeapDataType, MetricDirection, ConfusionMatrixValue
from code_loader.visualizers.default_visualizers import LeapImage
from code_loader.inner_leap_binder.leapbinder_decorators import (tensorleap_preprocess, tensorleap_gt_encoder,
                                                                 tensorleap_input_encoder, tensorleap_metadata,
                                                                 tensorleap_custom_visualizer)
from code_loader.contract.responsedataclasses import BoundingBox
from code_loader.contract.visualizer_classes import LeapImageWithBBox
from code_loader.utils import rescale_min_max

from ultralytics.utils import ops
from ultralytics.utils.plotting import output_to_target, Annotator  # doable
from ultralytics.utils.metrics import box_iou #doable
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
        ob_data_loader, ob_n_samples = create_data_with_ult(ob_cfg, ob_yolo_data, phase=phase)
        responses.append(
            PreprocessResponse(length=n_samples,
                               data={'dataloader':data_loader, "ob_dataloader": ob_data_loader},
                               state=dataset_type))
    return responses


# ------------------------------------------input and gt----------------------------------------------------------------

@tensorleap_input_encoder('image',channel_dim=1)
def input_encoder(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
    imgs, _, _,_, _, orig_shape =pre_process_dataloader(preprocess, idx, predictor)
    return imgs.astype('float32')


@tensorleap_gt_encoder('classes')
def gt_encoder(idx: int, preprocessing: PreprocessResponse) -> Union[
    ndarray[Any, dtype[Any]], tuple[ndarray[Any, dtype[Any]], Any]]:
    _, clss, bboxes, keypoints, _, _ = pre_process_dataloader(preprocessing, idx,predictor)
    if clss.shape[0]==0 and  bboxes.shape[0]==0:
        return np.full((1, 56), np.nan,dtype=np.float32)
    elif clss.shape[0]==0:
        temp_array=np.full((bboxes.shape[0], 56), np.nan,dtype=np.float32)
        temp_array[:,:4]=bboxes
        return temp_array
    elif bboxes.shape[0]==0:
        temp_array = np.full((clss.shape[0], 56), np.nan,dtype=np.float32)
        temp_array[:, 4] = clss
        return temp_array
    keypoints = keypoints.reshape(keypoints.shape[0],-1)
    concatenated = np.concatenate([bboxes,clss, keypoints],axis=1)
    return concatenated

# @tensorleap_gt_encoder('classes')
def ob_gt_encoder(idx: int, preprocessing: PreprocessResponse) -> np.ndarray:
    _, clss, bboxes, _, _ =pre_process_ob_dataloader(preprocessing, idx,predictor)
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
    nan_default_value = np.nan
    _, _, _, _, _, orig_shape = pre_process_dataloader(data, idx, predictor)
    gt_data = gt_encoder(idx, data)
    ob_gt_data = ob_gt_encoder(idx, data)
    cls_gt = np.expand_dims(gt_data[:, 4], axis=1)
    ob_cls_gt = np.expand_dims(ob_gt_data[:, 4], axis=1)
    bbox_gt = gt_data[:, :4]
    kpts = gt_data[:, 5:]
    kpts = kpts.reshape(kpts.shape[0], 17, 3)
    ob_clss_info = np.unique(ob_cls_gt, return_counts=True)
    pose_clss_info = np.unique(cls_gt, return_counts=True)
    num_preds = len(cls_gt) or nan_default_value

    count_dict = update_dict_count_cls(all_clss, pose_clss_info,nan_default_value)
    ob_count_dict = update_dict_count_cls(ob_all_clss, ob_clss_info,nan_default_value)
    if ob_count_dict["# 'person' class (0)"] is not None:
        pose_ob_people_ratio = float(count_dict["# 'person' class (0)"]/ob_count_dict["# 'person' class (0)"])
        pose_ob_people_diff = float(ob_count_dict["# 'person' class (0)"] - count_dict["# 'person' class (0)"])
        pose_ob_diff_to_total = float(pose_ob_people_diff/ob_count_dict["# 'person' class (0)"])
    else:
        pose_ob_people_ratio = nan_default_value
        pose_ob_people_diff = nan_default_value
        pose_ob_diff_to_total = nan_default_value
    areas, aspect_ratios = bbox_area_and_aspect_ratio(bbox_gt, data.data['dataloader'][idx]['resized_shape'])
    if len(cls_gt)>0:
        occlusion_matrix, areas_in_pixels, union_in_pixels = calculate_iou_all_pairs(bbox_gt, data.data['dataloader'][idx][
        'resized_shape'])
    else:
        occlusion_matrix, areas_in_pixels, union_in_pixels = nan_default_value, nan_default_value, nan_default_value
    no_nans_values = ~np.isnan(ob_clss_info[0]).any()
    pose_no_nans_values = ~np.isnan(pose_clss_info[0]).any() and len(pose_clss_info[0])
    sports = {1: 'bicycle', 3: 'motorcycle', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 34: 'baseball bat',
              35: 'baseball glove',
              36: 'skateboard',
              37: 'surfboard',
              38: 'tennis racket', }
    sport_count = sum([ob_clss_info[1][i] for i in range(len(ob_clss_info[1])) if ob_clss_info[0][i] in sports ])
    foods = { 60: 'dining table',  45: 'bowl',
                 46: 'banana',
                 47: 'apple',
                 48: 'sandwich',
                 49: 'orange',
                 50: 'broccoli',
                 51: 'carrot',
                 52: 'hot dog',
                 53: 'pizza',
                 54: 'donut',
                 55: 'cake',}
    foods_count = sum([ob_clss_info[1][i] for i in range(len(ob_clss_info[1])) if ob_clss_info[0][i] in foods])
    pets = { 15: 'cat',
            16: 'dog',}
    pets_count = sum([ob_clss_info[1][i] for i in range(len(ob_clss_info[1])) if ob_clss_info[0][i] in pets])
    animals = {14: 'bird',
                 15: 'cat',
                 16: 'dog',
                 17: 'horse',
                 18: 'sheep',
                 19: 'cow',
                 20: 'elephant',
                 21: 'bear',
                 22: 'zebra',
                 23: 'giraffe',}
    animals_count = sum([ob_clss_info[1][i] for i in range(len(ob_clss_info[1])) if ob_clss_info[0][i] in animals])
    vehicles = {1: 'bicycle',
             2: 'car',
             3: 'motorcycle',
             4: 'airplane',
             5: 'bus',
             6: 'train',
             7: 'truck',
             8: 'boat',}
    vehicles_count = sum([ob_clss_info[1][i] for i in range(len(ob_clss_info[1])) if ob_clss_info[0][i] in vehicles])
    d = {
        "image path": data.data['dataloader'].im_files[idx],
        "idx": idx,
        "# unique classes - ob": len(ob_clss_info[0]) if no_nans_values else nan_default_value,
        "# of objects - ob": int(ob_clss_info[1].sum()) if no_nans_values else nan_default_value,
        # Pose meta-data
        "% pose visible": (kpts[:,:,-1]==2).mean() if pose_no_nans_values else nan_default_value,
        "% pose all-vis people": ((kpts[:,:,-1]==2).mean(1)==1).sum() if pose_no_nans_values else nan_default_value,
        "num headless": ((kpts[:,:,-1][:,:5]==2).mean(1) == 0).sum() if pose_no_nans_values else nan_default_value, #num of people with no single head related keypoint
        "# of people with pose": float(num_preds) if pose_no_nans_values else float(0),
        "crowd": int(num_preds > 5) if pose_no_nans_values else int(0),
        # diffarence between object detection and pose labels
        "pose-ob people ratio": pose_ob_people_ratio,
        "pose-ob poeple diff": pose_ob_people_diff,
        "pose-ob diff to total": pose_ob_diff_to_total,
        # scene
        "# sport objects": sport_count,
        "sport": int(sport_count>0),
        "food": int(foods_count>0),
        "pet": int(pets_count>0),
        "animal": int(animals_count>0),
        "vehicle": int(vehicles_count>0),
        # Pose box meta-data
        "mean bbox area": float(areas.mean()) if pose_no_nans_values else nan_default_value,
        "var bbox area": float(areas.var()) if pose_no_nans_values else nan_default_value,
        "median bbox area": float(np.median(areas)) if pose_no_nans_values else nan_default_value,
        "max bbox area": float(np.max(areas)) if pose_no_nans_values else nan_default_value,
        "min bbox area": float(np.min(areas)) if pose_no_nans_values else nan_default_value,
        "bbox overlap": float(
            occlusion_matrix.sum() / areas_in_pixels.sum()) if pose_no_nans_values else nan_default_value,
        "max bbox overlap": float(
            (occlusion_matrix.sum(axis=1) / areas_in_pixels).max()) if pose_no_nans_values else nan_default_value,
        "orig_H": orig_shape[0],
        "orig_W": orig_shape[1],
    }

    # d.update(**count_dict)
    d.update(**ob_count_dict)
    return d


def preprocess_batch(batch):
    batch = predictor.preprocess(batch)
    if not isinstance(batch['ori_shape'], list):
        batch['ori_shape'] = [batch['ori_shape']]
    if not isinstance(batch['ratio_pad'], list):
        batch['ratio_pad'] = [batch['ratio_pad']]
    batch['img'] = batch["img"].unsqueeze(0)
    return batch

def postprocess(pred: np.ndarray,
                                feat80: np.ndarray, feat40: np.ndarray, feat20: np.ndarray,
                                kpts: np.ndarray):
    all_feats = [pred, feat80, feat40, feat20, kpts]
    y = [torch.from_numpy(copy.deepcopy(t)) for t in all_feats]
    y = [y[0], ([y[1:4]], y[4])]
    preds = predictor.postprocess(y)
    return preds
# ----------------------------------------------------------loss--------------------------------------------------------

@tensorleap_custom_loss(name="total_loss")
def loss(exm_pred,pred8_0,pred40,pred20, keypoints_pred, gt):
    # return np.zeros(1)
    d={}
    d["bboxes"] = torch.from_numpy(gt[...,:4]).squeeze(0)
    d["cls"] = torch.from_numpy(gt[...,4])
    keypoints = torch.from_numpy(gt.squeeze(0)[...,5:]).reshape(gt[...,5].shape[1], 17, 3)
    d['keypoints'] = keypoints
    d["batch_idx"] = torch.zeros_like(d['cls'])
    y_pred_torch = [torch.from_numpy(s) for s in [pred8_0,pred40,pred20]]
    y_pred_torch = [y_pred_torch, torch.from_numpy(keypoints_pred)]
    all_loss,parts= criterion(y_pred_torch, d) # box, cls, dfl, kpt_location, kpt_visibility
    return all_loss.unsqueeze(0).numpy()


# ------------------------------------------------------visualizers-----------------------------------------------------


@tensorleap_custom_visualizer('image_visualizer', LeapDataType.Image)
def image_visualizer(image: np.ndarray) -> LeapImage:
    image = rescale_min_max(image.squeeze(0))
    return LeapImage((image.transpose(1,2,0)), compress=False)


def ensure_size(img_bgr, expected_w: int, expected_h: int):
    """
    Return image resized to (expected_w, expected_h) if needed.
    img_bgr: numpy array (H,W,3) in BGR
    expected_w/expected_h: original image size used for labels/preds
    """
    if img_bgr is None:
        return img_bgr
    h, w = img_bgr.shape[:2]
    if (w, h) != (expected_w, expected_h):
        # Resize to original pixel size so overlays/coords stay correct
        import cv2
        img_bgr = cv2.resize(img_bgr, (expected_w, expected_h), interpolation=cv2.INTER_LINEAR)
    return img_bgr



@tensorleap_custom_visualizer('gt_keypoints', LeapDataType.Image)
def gt_visualizer(image: np.ndarray,
                  gt: np.ndarray,
                  data : SamplePreprocessResponse) -> LeapImage:
    meta_data = metadata_per_img(int(data.sample_ids), data.preprocess_response)
    image = rescale_min_max(image)
    image = np.transpose(image.squeeze(0), [1, 2, 0])
    keypoints = gt[...,5:][0,::]
    # turn gt coordinates to image coordinates
    bboxes = gt[...,:5][0,::]
    imgsz = image.shape[:2]
    bboxes = ops.xywh2xyxy(bboxes[::,:4]) * np.array(imgsz)[[1, 0, 1, 0]]   # target boxes
    pred_kpts = torch.from_numpy(keypoints.reshape(len(keypoints), 17, 3)) \
        if len(keypoints[0,::]) else keypoints
    h, w = imgsz
    pred_kpts[..., 0] *= w
    pred_kpts[..., 1] *= h
    if not np.isfinite(gt).any():
        return LeapImage(image)
    annotator = Annotator(
        np.ascontiguousarray(image),  # Classify tasks default to pil=True
        example={0:'person'},
    )
    annotator.lw = 1
    # Plot Pose results
    if pred_kpts is not None:
        for i, k in enumerate(reversed(pred_kpts)):
            annotator.kpts(
                k,
                (meta_data['orig_W'], meta_data['orig_H']),
                radius=5,
                kpt_line=True,
                kpt_color=None,
            )
    bboxes = np.insert(bboxes, 4, 1, axis=1) # add conf
    bboxes = np.insert(bboxes, 5, 0, axis=1)  # add conf
    pred_boxes = Boxes(bboxes, (meta_data['orig_W'], meta_data['orig_H']))
    names = {0: 'person'}
    is_obb = False
    conf = True
    for i, d in enumerate(reversed(pred_boxes)):
        c, conf, id = int(d.cls), float(d.conf) if conf else None, None if d.id is None else int(d.id.item())
        name = ("" if id is None else f"id:{id} ") + names[c]
        label = 'person'
        box = d.xyxyxyxy.reshape(-1, 4, 2).squeeze() if is_obb else d.xyxy.squeeze()
        annotator.box_label(
            box,
            label,
            None,
            rotated=is_obb,
        )
    im = annotator.result()
    return LeapImage(im, compress=False)


@tensorleap_custom_visualizer('keypoints_visualizer', LeapDataType.Image)
def pred_visualizer(image: np.ndarray,
                  pred: np.ndarray, feat80: np.ndarray, feat40: np.ndarray, feat20: np.ndarray, kpts: np.ndarray,
                  data : SamplePreprocessResponse) -> LeapImage:
    meta_data = metadata_per_img(int(data.sample_ids), data.preprocess_response)
    image = rescale_min_max(image)
    image = np.transpose(image.squeeze(0), [1, 2, 0])
    y_pred = postprocess(pred, feat80, feat40, feat20, kpts)
    y_pred = y_pred[0]
    pred_kpts = y_pred[:,6:].view(len(y_pred), 17, 3) if len(y_pred) else y_pred
    annotator = Annotator(
        np.ascontiguousarray(image),  # Classify tasks default to pil=True
        example={0:'person'},
    )
    annotator.lw = 1
    # Plot Pose results
    if pred_kpts is not None:
        for i, k in enumerate(reversed(pred_kpts)):
            annotator.kpts(
                k,
                (meta_data['orig_W'], meta_data['orig_H']),
                radius=5,
                kpt_line=True,
                kpt_color=None,
            )

    pred_boxes = Boxes(y_pred[:,:6], (meta_data['orig_W'], meta_data['orig_H']))
    names = {0: 'person'}
    is_obb = False
    conf = True
    for i, d in enumerate(reversed(pred_boxes)):
        c, conf, id = int(d.cls), float(d.conf) if conf else None, None if d.id is None else int(d.id.item())
        name = ("" if id is None else f"id:{id} ") + names[c]
        label = 'person'
        box = d.xyxyxyxy.reshape(-1, 4, 2).squeeze() if is_obb else d.xyxy.squeeze()
        annotator.box_label(
            box,
            label,
            None,
            rotated=is_obb,
        )
    im = annotator.result()
    return LeapImage(im, compress=False)

@tensorleap_custom_metric("cost", direction=MetricDirection.Downward)
def cost(pred80,pred40,pred20,keypoints_pred, gt):
    # return np.zeros(1)
    d={}
    d["bboxes"] = torch.from_numpy(gt[...,:4]).squeeze(0)
    d["cls"] = torch.from_numpy(gt[...,4])
    keypoints = torch.from_numpy(gt.squeeze(0)[...,5:]).reshape(gt[...,5].shape[1], 17, 3)
    d['keypoints'] = keypoints
    d["batch_idx"] = torch.zeros_like(d['cls'])
    y_pred_torch = [torch.from_numpy(s) for s in [pred80,pred40,pred20]]
    y_pred_torch = [y_pred_torch, torch.from_numpy(keypoints_pred)]
    _,loss_parts = criterion(y_pred_torch, d) # loss(box, pose, kobj, cls, dfl)
    # Find stats of cases where image with no persons was detected as having a person

    return {"box":loss_parts[0].unsqueeze(0).numpy(), "pose": loss_parts[1].unsqueeze(0).numpy(),
            "kobj": loss_parts[2].unsqueeze(0).numpy(), "cls":loss_parts[3].unsqueeze(0).numpy(),
            "dfl":loss_parts[4].unsqueeze(0).numpy()}



@tensorleap_custom_metric('Matrices', direction={'precision(B)': MetricDirection.Upward, 'recall(B)': MetricDirection.Upward, 'mAP50(B)': MetricDirection.Upward,
             'mAP50-95(B)': MetricDirection.Upward, 'precision(P)': MetricDirection.Upward, 'recall(P)': MetricDirection.Upward,
             'mAP50(P)': MetricDirection.Upward, 'mAP50-95(P)': MetricDirection.Upward, 'fitness': MetricDirection.Upward })
def get_matrices(pred: np.ndarray, feat80: np.ndarray, feat40: np.ndarray, feat20: np.ndarray, kpts: np.ndarray,
                 preprocess : SamplePreprocessResponse):
    default_value = np.ones(1) * np.nan
    batch = preprocess.preprocess_response.data['dataloader'][int(preprocess.sample_ids)]
    batch = preprocess_batch(batch)
    preds = postprocess(pred, feat80, feat40, feat20, kpts)
    stats = {'precision(B)': default_value, 'recall(B)': default_value, 'mAP50(B)': default_value,
             'mAP50-95(B)': default_value, 'precision(P)': default_value, 'recall(P)': default_value,
             'mAP50(P)': default_value, 'mAP50-95(P)': default_value, 'fitness': default_value, 'FP_human': default_value,}

    if all(len(preds[i]) == 0 for i in range(len(preds))):
        return stats
    predictor.update_metrics(preds, batch)
    stats = predictor.get_stats()
    stats = {key.split('/')[-1]: np.array([float(value)]) for key, value in stats.items()}
    return stats
