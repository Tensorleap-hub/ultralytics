import os
import json
from pathlib import Path
import torch
from code_loader.inner_leap_binder.leapbinder_decorators import tensorleap_custom_loss, tensorleap_custom_metric, \
    tensorleap_instances_length_encoder, tensorleap_custom_instances_metric
from ultralytics.tensorleap_folder.global_params import cfg, yolo_data, criterion, all_clss,possible_float_like_nan_types,wanted_cls_dic, predictor
from ultralytics.tensorleap_folder.utils import create_data_with_ult, pre_process_dataloader, \
    update_dict_count_cls, bbox_area_and_aspect_ratio, calculate_iou_all_pairs, get_dataset_split
from typing import List, Dict, Union
import numpy as np
from code_loader.contract.datasetclasses import PreprocessResponse, DataStateType, SamplePreprocessResponse, \
    ConfusionMatrixElement, ElementInstance
from code_loader.contract.enums import LeapDataType, MetricDirection, ConfusionMatrixValue
from code_loader.visualizers.default_visualizers import LeapImage
from code_loader.inner_leap_binder.leapbinder_decorators import (tensorleap_preprocess,
                                                                 tensorleap_element_instance_preprocess,
                                                                 tensorleap_gt_encoder,
                                                                 tensorleap_input_encoder, tensorleap_metadata,
                                                                 tensorleap_custom_visualizer,
                                                                 tensorleap_unlabeled_preprocess,
                                                                 tensorleap_instances_masks_encoder)
from code_loader.contract.responsedataclasses import BoundingBox
from code_loader.contract.visualizer_classes import LeapImageWithBBox
from code_loader.utils import rescale_min_max
from ultralytics.utils.plotting import output_to_target
from ultralytics.utils.metrics import box_iou


# ----------------------------------------------------aggressor map-----------------------------------------------------

_AGG_MAP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "aggressor_map.json")
try:
    with open(_AGG_MAP_PATH) as _f:
        AGGRESSOR_MAP = json.load(_f)
except FileNotFoundError:
    AGGRESSOR_MAP = {}


def _base_idx(idx):
    """Resolve a sample id to its int dataloader index.

    The element-instance preprocess uses str sample ids and adds synthetic
    per-instance ids of the form '<sample_id>_<instance_id>'. Either form (and a
    plain int) resolves to the underlying image index."""
    return int(str(idx).split('_')[0])


# --- evaluation class merge ------------------------------------------------------------
# The model is 20-class (oven=5, microwave=6 SEPARATE), but oven<->microwave is the same
# concept ("oven_like") and their confusion must NOT count as a class error in metrics.
# Canonicalize microwave(6) -> oven(5) wherever class identity is compared (loss, cost,
# per-instance same-class matching) and display the merged class as "oven_like".
# Set EVAL_CLASS_MERGE = {} to disable and score the raw 20 classes.
EVAL_CLASS_MERGE = {6: 5}            # {microwave: oven}
_MERGE_LABEL = "oven_like"


def _merge_eval_cls(cls):
    """Map merged class ids to their canonical id. Accepts a torch tensor or numpy/array."""
    if hasattr(cls, "clone"):                 # torch tensor
        out = cls.clone()
        for src, dst in EVAL_CLASS_MERGE.items():
            out[out == src] = dst
        return out
    out = np.asarray(cls).copy()
    for src, dst in EVAL_CLASS_MERGE.items():
        out[out == src] = dst
    return out


def _eval_cls_name(idx):
    """Class display name after the eval merge (merged classes show as the merged label)."""
    idx = int(idx)
    if idx in EVAL_CLASS_MERGE:
        idx = EVAL_CLASS_MERGE[idx]
    if idx in EVAL_CLASS_MERGE.values():
        return _MERGE_LABEL
    return all_clss.get(idx, "Unknown Class")


def _finite_or_none(d):
    """JSON / Elasticsearch cannot represent NaN or Infinity. The streaming-handler
    bulk-indexes metadata documents and crash-loops if any value serializes as the
    non-standard token 'NaN' (e.g. an overlap ratio divided by a zero-area gt box).
    Replace any non-finite numeric value with None (indexed as null), matching how the
    per-class counts already represent 'no value'."""
    return {k: (None if isinstance(v, (float, np.floating)) and not np.isfinite(v) else v)
            for k, v in d.items()}

# ----------------------------------------------------data processing---------------------------------------------------

@tensorleap_instances_masks_encoder('image')
def instance_mask_encoder(idx: str, preprocess: PreprocessResponse, instance_idx) -> ElementInstance:
    gt = gt_encoder(idx, preprocess)
    label = gt[instance_idx]
    mask = np.zeros((3, 640, 640))
    x, y, w, h, label_id = label
    if np.isnan([x, y, w, h, label_id]).any():
        return None
    img_width, img_height = mask.shape[1], mask.shape[2]
    x, y, w, h = round(x * img_width - ((w * img_width) / 2)), round(y * img_height - ((h * img_height) / 2)), round(w * img_width), round(h * img_height)

    mask[:, y:y+h, x:x+w] = 1

    element_instance = ElementInstance(all_clss.get(int(label_id), "Unknown Class"), mask)

    return element_instance


@tensorleap_instances_length_encoder('image')
def instances_length_encoder(idx: str, preprocess: PreprocessResponse) -> int:
    # Defensive: this runs for every sample at parse time. If a sample id ever exceeds the
    # built dataset length (e.g. a stale .cache yielding fewer labels than the txt count),
    # report no instances instead of crashing the whole dataset parse with an IndexError.
    if _base_idx(idx) >= len(preprocess.data['dataloader']):
        return 0
    gt = gt_encoder(idx, preprocess)
    for label in gt:
        x, y, w, h, label_id = label
        if np.isnan([x, y, w, h, label_id]).any():
            return 0
    return len(gt)


@tensorleap_element_instance_preprocess(instances_length_encoder, instance_mask_encoder)
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
        sample_ids = [str(idd) for idd in range(n_samples)]
        responses.append(
            PreprocessResponse(sample_ids=sample_ids,
                               data={'dataloader':data_loader},
                               sample_id_type=str,
                               state=dataset_type))
    return responses


# ------------------------------------------input and gt----------------------------------------------------------------

@tensorleap_input_encoder('image', channel_dim=1)
def input_encoder(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
    imgs, _, _,_=pre_process_dataloader(preprocess, idx, predictor)
    return imgs.astype('float32')


@tensorleap_gt_encoder('classes')
def gt_encoder(idx: int, preprocessing: PreprocessResponse) -> np.ndarray:
    if preprocessing.state == DataStateType.unlabeled:
        return np.full((1, 5), np.nan,dtype=np.float32)
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


@tensorleap_metadata("image info a", metadata_type=possible_float_like_nan_types)
def metadata_per_img(idx: int, data: PreprocessResponse) -> Dict[str, Union[str, int, float]]:
    nan_default_value = None
    gt_data = gt_encoder(idx, data)
    bidx = _base_idx(idx)
    cls_gt = np.expand_dims(gt_data[:, 4], axis=1)
    bbox_gt = gt_data[:, :4]
    clss_info = np.unique(cls_gt, return_counts=True)
    count_dict = update_dict_count_cls(all_clss, clss_info,nan_default_value)
    areas, aspect_ratios = bbox_area_and_aspect_ratio(bbox_gt, data.data['dataloader'][bidx]['resized_shape'])
    occlusion_matrix, areas_in_pixels, union_in_pixels = calculate_iou_all_pairs(bbox_gt, data.data['dataloader'][bidx][
        'resized_shape'])
    no_nans_values = ~np.isnan(clss_info[0]).any()
    d = {
        "image path": data.data['dataloader'].im_files[bidx],
        "idx": bidx,
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
    return _finite_or_none(d)


@tensorleap_metadata("aggressor")
def metadata_aggressor(idx, data) -> Dict[str, Union[str, int, float]]:
    stem = Path(data.data['dataloader'].im_files[_base_idx(idx)]).stem
    info = AGGRESSOR_MAP.get(stem, {"family": "none", "axis": "none", "role": "clean"})
    return {"aggressor_family": info.get("family", "none"),
            "aggressor_axis": info.get("axis", "none"),
            "aggressor_role": info.get("role", "clean"),
            "is_aggressor": int(info.get("role") == "aggressor")}



# ----------------------------------------------------------loss--------------------------------------------------------

@tensorleap_custom_loss("total_loss")
def loss(pred80,pred40,pred20,gt,demo_pred):
    gt=np.squeeze(gt,axis=0)
    d={}
    d["bboxes"] = torch.from_numpy(gt[...,:4])
    d["cls"] = torch.from_numpy(_merge_eval_cls(gt[...,4]))
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


@tensorleap_custom_visualizer('image_visualizer_original', LeapDataType.Image)
def image_visualizer_original(image: np.ndarray, preprocess: SamplePreprocessResponse) -> LeapImage:
    """Element-instance visualizer: show the ORIGINAL image for an instance element.
    Resolves the instance id '<sample>_<inst>' back to its base sample via the
    preprocess mapping, then re-encodes that image (mirrors the reference repo)."""
    sid = str(np.asarray(preprocess.sample_ids).reshape(-1)[0])
    mapping = getattr(preprocess.preprocess_response, 'instance_to_sample_ids_mappings', None)
    base_id = mapping.get(sid, sid) if mapping else sid
    img = input_encoder(base_id, preprocess.preprocess_response)
    img = rescale_min_max(img)
    if img.ndim == 4:                  # auto-batched when called inside the integration test
        img = img[0]
    return LeapImage(img.transpose(1, 2, 0), compress=False)


@tensorleap_custom_visualizer("bb_decoder", LeapDataType.ImageWithBBox)
def bb_decoder(image: np.ndarray, predictions: np.ndarray) -> LeapImageWithBBox:
    image=image.squeeze(0)
    y_pred = predictor.postprocess(torch.from_numpy(predictions.copy()))
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
    batch = preprocess.preprocess_response.data['dataloader'][_base_idx(np.asarray(preprocess.sample_ids).reshape(-1)[0])]
    batch["imgsz"]     = (batch["resized_shape"],)
    batch["ori_shape"] = (batch["ori_shape"],)
    batch["ratio_pad"] = (batch["ratio_pad"],)
    batch["img"]       = batch["img"].unsqueeze(0)
    pred = predictor.postprocess(torch.from_numpy(y_pred.copy()))[0]
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
    d["cls"] = torch.from_numpy(_merge_eval_cls(gt[...,4]))
    d["batch_idx"] = torch.zeros_like(d['cls'])
    y_pred_torch = [torch.from_numpy(s) for s in [pred80,pred40,pred20]]
    _,loss_parts= criterion(y_pred_torch, d)
    return {"box":loss_parts[0].unsqueeze(0).numpy(),"cls":loss_parts[1].unsqueeze(0).numpy(),"dfl":loss_parts[2].unsqueeze(0).numpy()}


@tensorleap_custom_metric('Confusion Matrix', direction=MetricDirection.Downward)
def confusion_matrix_metric(y_pred: np.ndarray, preprocess: SamplePreprocessResponse):
    threshold=cfg.iou
    confusion_matrix_elements = []
    batch=preprocess.preprocess_response.data['dataloader'][_base_idx(np.asarray(preprocess.sample_ids).reshape(-1)[0])]
    batch["imgsz"]=(batch["resized_shape"],)
    batch["ori_shape"]=(batch["ori_shape"],)
    batch["ratio_pad"]= (batch["ratio_pad"],)
    batch["img"]=batch["img"].unsqueeze(0)
    pred = predictor.postprocess(torch.from_numpy(y_pred.copy()))[0]
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
            class_name = _eval_cls_name(gt_idx)
            gt_label = f"{class_name}"
            confidence = predn[i, 4]
            if prediction:  # TP
                confusion_matrix_elements.append(ConfusionMatrixElement(
                    str(gt_label),
                    ConfusionMatrixValue.Positive,
                    float(confidence)
                ))
            else:  # FP
                class_name = _eval_cls_name(int(predn[i,5]))
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
            class_name = _eval_cls_name(int(label_idx))
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


# ----------------------------------------------------instance metrics--------------------------------------------------

def _resolve_instance_sample_id(preprocess: SamplePreprocessResponse):
    """Return the underlying base sample id (str) for an instance metric call,
    translating an instance id ('k_i') via the preprocess mapping when present."""
    id_ = str(np.asarray(preprocess.sample_ids).reshape(-1)[0])  # coerce numpy.str_ -> str
    mapping = getattr(preprocess.preprocess_response, 'instance_to_sample_ids_mappings', None)
    return mapping.get(id_, id_) if mapping else id_


@tensorleap_custom_instances_metric("instance_best_iou", direction=MetricDirection.Upward)
def instance_best_iou(y_pred: np.ndarray, preprocess: SamplePreprocessResponse):
    """Per-GT-instance IoU of the best-matching same-class prediction (0 if unmatched)."""
    sample_id = _resolve_instance_sample_id(preprocess)
    n_instances = instances_length_encoder(sample_id, preprocess.preprocess_response)
    result = {i: np.zeros(1, dtype=np.float32) for i in range(n_instances)}
    if n_instances == 0:
        return result

    batch = preprocess.preprocess_response.data['dataloader'][_base_idx(sample_id)]
    batch["imgsz"] = (batch["resized_shape"],)
    batch["ori_shape"] = (batch["ori_shape"],)
    batch["ratio_pad"] = (batch["ratio_pad"],)
    batch["img"] = batch["img"].unsqueeze(0)
    pred = predictor.postprocess(torch.from_numpy(y_pred.copy()))[0]
    predictor.seen = 0
    predictor.args.plots = False
    predictor.stats = {'tp': []}
    pbatch = predictor._prepare_batch(0, batch)
    gt_cls, gt_bbox = pbatch.pop("cls"), pbatch.pop("bbox")
    predn = predictor._prepare_pred(pred, pbatch)

    if predn.shape[0] == 0 or gt_bbox.shape[0] == 0:
        return result

    iou_mat = box_iou(gt_bbox, predn[:, :4])
    same_class = (_merge_eval_cls(gt_cls).view(-1, 1) == _merge_eval_cls(predn[:, 5]).view(1, -1))   # match only same-class preds
    iou_mat = iou_mat * same_class.to(iou_mat.dtype)

    best_iou_per_gt = iou_mat.max(dim=1).values.numpy()
    for i in range(min(n_instances, best_iou_per_gt.shape[0])):
        result[i] = np.array([best_iou_per_gt[i]], dtype=np.float32)
    return result


@tensorleap_custom_instances_metric("instance_match_confidence", direction=MetricDirection.Upward)
def instance_match_confidence(y_pred: np.ndarray, preprocess: SamplePreprocessResponse):
    """Per-GT-instance confidence of the best-matching same-class prediction (0 if unmatched)."""
    sample_id = _resolve_instance_sample_id(preprocess)
    n_instances = instances_length_encoder(sample_id, preprocess.preprocess_response)
    result = {i: np.zeros(1, dtype=np.float32) for i in range(n_instances)}
    if n_instances == 0:
        return result

    batch = preprocess.preprocess_response.data['dataloader'][_base_idx(sample_id)]
    batch["imgsz"] = (batch["resized_shape"],)
    batch["ori_shape"] = (batch["ori_shape"],)
    batch["ratio_pad"] = (batch["ratio_pad"],)
    batch["img"] = batch["img"].unsqueeze(0)
    pred = predictor.postprocess(torch.from_numpy(y_pred.copy()))[0]
    predictor.seen = 0
    predictor.args.plots = False
    predictor.stats = {'tp': []}
    pbatch = predictor._prepare_batch(0, batch)
    gt_cls, gt_bbox = pbatch.pop("cls"), pbatch.pop("bbox")
    predn = predictor._prepare_pred(pred, pbatch)

    if predn.shape[0] == 0 or gt_bbox.shape[0] == 0:
        return result

    iou_mat = box_iou(gt_bbox, predn[:, :4])
    same_class = (_merge_eval_cls(gt_cls).view(-1, 1) == _merge_eval_cls(predn[:, 5]).view(1, -1))
    iou_mat = iou_mat * same_class.to(iou_mat.dtype)

    best_pred_idx = iou_mat.argmax(dim=1).numpy()
    best_iou_per_gt = iou_mat.max(dim=1).values.numpy()
    confidences = predn[:, 4].numpy()
    for i in range(min(n_instances, best_pred_idx.shape[0])):
        if best_iou_per_gt[i] > 0:
            result[i] = np.array([confidences[best_pred_idx[i]]], dtype=np.float32)
    return result


# ---------------------------------------------------------main------------------------------------------------------


