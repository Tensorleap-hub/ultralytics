import numpy as np
import torch
from code_loader.contract.datasetclasses import ConfusionMatrixElement, SamplePreprocessResponse
from code_loader.contract.enums import ConfusionMatrixValue, MetricDirection
from code_loader.inner_leap_binder.leapbinder_decorators import (
    tensorleap_custom_loss,
    tensorleap_custom_metric,
)
from ultralytics.utils.metrics import box_iou

from ultralytics.tensorleap_folder.global_params import (
    all_clss,
    cfg,
    criterion,
    predictor,
    wanted_cls_dic,
)


def _prepare_detection_batch(y_pred: np.ndarray, preprocess: SamplePreprocessResponse):
    batch = preprocess.preprocess_response.data["dataloader"][int(preprocess.sample_ids)]
    batch["imgsz"] = (batch["resized_shape"],)
    batch["ori_shape"] = (batch["ori_shape"],)
    batch["ratio_pad"] = (batch["ratio_pad"],)
    batch["img"] = batch["img"].unsqueeze(0)
    pred = predictor.postprocess(torch.from_numpy(y_pred.copy()))[0]
    predictor.seen = 0
    predictor.args.plots = False
    predictor.stats = {"tp": []}
    pbatch = predictor._prepare_batch(0, batch)
    cls = pbatch.pop("cls")
    bbox = pbatch.pop("bbox")
    predn = predictor._prepare_pred(pred, pbatch)
    return batch, cls, bbox, predn


def _match_detections(predn: torch.Tensor, cls: torch.Tensor, bbox: torch.Tensor, threshold: float):
    n_pred = len(predn)
    n_gt = len(cls)
    matched_gt = np.zeros(n_gt, dtype=bool)
    pred_gt_match = np.full(n_pred, -1, dtype=int)
    pred_is_tp = np.zeros(n_pred, dtype=bool)
    gt_is_detected = np.zeros(n_gt, dtype=bool)

    if n_pred == 0 or n_gt == 0:
        return pred_gt_match, pred_is_tp, gt_is_detected

    iou_matrix = box_iou(predn[:, :4], bbox).numpy()
    pred_classes = predn[:, 5].cpu().numpy().astype(int)
    gt_classes = cls.cpu().numpy().astype(int)

    candidate_matches = np.argwhere(iou_matrix >= threshold)
    if len(candidate_matches) == 0:
        return pred_gt_match, pred_is_tp, gt_is_detected

    match_scores = iou_matrix[candidate_matches[:, 0], candidate_matches[:, 1]]
    sorted_candidates = candidate_matches[np.argsort(-match_scores)]
    used_pred = np.zeros(n_pred, dtype=bool)

    for pred_idx, gt_idx in sorted_candidates:
        if used_pred[pred_idx] or matched_gt[gt_idx]:
            continue
        used_pred[pred_idx] = True
        matched_gt[gt_idx] = True
        pred_gt_match[pred_idx] = gt_idx
        gt_is_detected[gt_idx] = True
        pred_is_tp[pred_idx] = pred_classes[pred_idx] == gt_classes[gt_idx]

    return pred_gt_match, pred_is_tp, gt_is_detected


@tensorleap_custom_loss("total_loss")
def loss(pred80, pred40, pred20, gt, demo_pred):
    gt = np.squeeze(gt, axis=0)
    target = {}
    target["bboxes"] = torch.from_numpy(gt[..., :4])
    target["cls"] = torch.from_numpy(gt[..., 4])
    target["batch_idx"] = torch.zeros_like(target["cls"])
    y_pred_torch = [torch.from_numpy(s) for s in [pred80, pred40, pred20]]
    all_loss, _ = criterion(y_pred_torch, target)
    return all_loss.unsqueeze(0).numpy()


@tensorleap_custom_metric(
    "ious",
    direction={
        **{class_name: MetricDirection.Upward for class_name in wanted_cls_dic.keys()},
        "mean sample iou": MetricDirection.Upward,
    },
)
def ious(y_pred: np.ndarray, preprocess: SamplePreprocessResponse):
    default_value = np.ones(1) * -1
    _, cls_gt, boxes_gt, predn = _prepare_detection_batch(y_pred, preprocess)
    wanted_mask = np.isin(cls_gt.numpy(), np.array(list(wanted_cls_dic.values())))
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
    mean_iou_sample = np.expand_dims(all_instance_ious.mean(), axis=0)
    for c_id, c_name in wanted_cls_dic.items():
        mask_c = (cls_gt.numpy() == c_name) & wanted_mask
        if mask_c.any():
            iou_dic[c_id] = np.expand_dims(assigned_iou_per_gt[mask_c].mean(), axis=0)

    iou_dic["mean sample iou"] = mean_iou_sample
    return iou_dic


@tensorleap_custom_metric("cost", direction=MetricDirection.Downward)
def cost(pred80, pred40, pred20, gt):
    gt = np.squeeze(gt, axis=0)
    target = {}
    target["bboxes"] = torch.from_numpy(gt[..., :4])
    target["cls"] = torch.from_numpy(gt[..., 4])
    target["batch_idx"] = torch.zeros_like(target["cls"])
    y_pred_torch = [torch.from_numpy(s) for s in [pred80, pred40, pred20]]
    _, loss_parts = criterion(y_pred_torch, target)
    return {
        "box": loss_parts[0].unsqueeze(0).numpy(),
        "cls": loss_parts[1].unsqueeze(0).numpy(),
        "dfl": loss_parts[2].unsqueeze(0).numpy(),
    }


SCORE_DIRECTIONS = {
    **{
        f"{metric}({class_name})": MetricDirection.Upward
        for class_name in all_clss.values()
        for metric in ("precision", "recall", "f1")
    },
    **{f"{metric}(global)": MetricDirection.Upward for metric in ("precision", "recall", "f1")},
}


@tensorleap_custom_metric("Detection Scores", direction=SCORE_DIRECTIONS)
def detection_scores(y_pred: np.ndarray, preprocess: SamplePreprocessResponse):
    default_value = np.ones(1) * np.nan
    _, cls, bbox, predn = _prepare_detection_batch(y_pred, preprocess)
    threshold = cfg.iou
    class_names = list(all_clss.values())
    scores = {
        f"{metric}({class_name})": default_value.copy()
        for class_name in class_names
        for metric in ("precision", "recall", "f1")
    }
    scores.update({f"{metric}(global)": default_value.copy() for metric in ("precision", "recall", "f1")})

    pred_gt_match, pred_is_tp, gt_is_detected = _match_detections(predn, cls, bbox, threshold)
    pred_classes = predn[:, 5].cpu().numpy().astype(int) if len(predn) else np.array([], dtype=int)
    gt_classes = cls.cpu().numpy().astype(int) if len(cls) else np.array([], dtype=int)

    total_tp = int(pred_is_tp.sum())
    total_fp = int(len(predn) - total_tp)
    total_fn = int(len(cls) - total_tp)
    global_precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else np.nan
    global_recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else np.nan
    global_f1 = (
        2 * global_precision * global_recall / (global_precision + global_recall)
        if not np.isnan(global_precision) and not np.isnan(global_recall) and (global_precision + global_recall)
        else np.nan
    )
    scores["precision(global)"] = np.array([global_precision])
    scores["recall(global)"] = np.array([global_recall])
    scores["f1(global)"] = np.array([global_f1])

    for class_idx, class_name in all_clss.items():
        class_pred_mask = pred_classes == class_idx
        class_gt_mask = gt_classes == class_idx
        class_tp = int(np.sum(pred_is_tp & class_pred_mask))
        class_fp = int(np.sum(class_pred_mask & ~pred_is_tp))
        class_fn = int(np.sum(class_gt_mask & ~gt_is_detected))

        precision = class_tp / (class_tp + class_fp) if class_tp + class_fp else np.nan
        recall = class_tp / (class_tp + class_fn) if class_tp + class_fn else np.nan
        f1 = 2 * precision * recall / (precision + recall) if not np.isnan(precision) and not np.isnan(recall) and (precision + recall) else np.nan

        scores[f"precision({class_name})"] = np.array([precision])
        scores[f"recall({class_name})"] = np.array([recall])
        scores[f"f1({class_name})"] = np.array([f1])

    return scores


@tensorleap_custom_metric("Confusion Matrix", direction=MetricDirection.Downward)
def confusion_matrix_metric(y_pred: np.ndarray, preprocess: SamplePreprocessResponse):
    threshold = cfg.iou
    confusion_matrix_elements = []
    batch, cls, bbox, predn = _prepare_detection_batch(y_pred, preprocess)
    if len(predn) != 0:
        ious_array = box_iou(bbox, predn[:, :4]).numpy().T
        prediction_detected = np.any((ious_array > threshold), axis=1)
        max_iou_ind = np.argmax(ious_array, axis=1)
        for i, prediction in enumerate(prediction_detected):
            gt_idx = int(batch["cls"][max_iou_ind[i]])
            class_name = all_clss.get(gt_idx)
            gt_label = f"{class_name}"
            confidence = predn[i, 4]
            if prediction:
                confusion_matrix_elements.append(
                    ConfusionMatrixElement(
                        str(gt_label),
                        ConfusionMatrixValue.Positive,
                        float(confidence),
                    )
                )
            else:
                class_name = all_clss.get(int(predn[i, 5]))
                pred_label = f"{class_name}"
                confusion_matrix_elements.append(
                    ConfusionMatrixElement(
                        str(pred_label),
                        ConfusionMatrixValue.Negative,
                        float(confidence),
                    )
                )
    else:
        ious_array = np.zeros((1, cls.shape[0]))
    gts_detected = np.any((ious_array > threshold), axis=0)
    for k, gt_detection in enumerate(gts_detected):
        label_idx = cls[k]
        if not gt_detection:
            class_name = all_clss.get(int(label_idx))
            confusion_matrix_elements.append(
                ConfusionMatrixElement(
                    f"{class_name}",
                    ConfusionMatrixValue.Positive,
                    float(0),
                )
            )
    if all(~gts_detected):
        confusion_matrix_elements.append(
            ConfusionMatrixElement(
                "background",
                ConfusionMatrixValue.Positive,
                float(0),
            )
        )
    return [confusion_matrix_elements]
