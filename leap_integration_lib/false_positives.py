from typing import Dict, List

import numpy as np
import torch
from code_loader.contract.datasetclasses import (
    DataStateType,
    ElementInstance,
    PreprocessResponse,
    SamplePreprocessResponse,
)
from code_loader.contract.enums import DatasetMetadataType
from code_loader.inner_leap_binder.leapbinder_decorators import (
    tensorleap_false_positive_custom_latent_space,
    tensorleap_false_positive_empty_gt,
    tensorleap_false_positive_instances_encoder,
)
from ultralytics.utils.metrics import box_iou

from leap_integration_lib.data import (
    INSTANCE_METADATA_TYPES,
    MAX_GT_BOXES,
    _bbox_appearance_vector,
    input_encoder,
)
from leap_integration_lib.metrics import _match_detections, _prepare_detection_batch
from ultralytics.tensorleap_folder.global_params import all_clss, cfg, predictor

# The prediction the engine hands over is keyed by its prediction-type name (leap_integration.py).
DETECTION_PREDICTION = "object detection"

# The GT instances' columns, plus what only a false positive has. GT rows carry these extra keys as
# absent; FP rows carry the GT keys computed from the predicted box, so both populations share them.
FALSE_POSITIVE_METADATA_TYPES = {
    **INSTANCE_METADATA_TYPES,
    "fp_confidence": DatasetMetadataType.float,
    "fp_type": DatasetMetadataType.string,
}


def false_positive_boxes(sample_id: str, preprocess: PreprocessResponse,
                         detection_pred: np.ndarray, width: int, height: int) -> List[dict]:
    """The image's false-positive detections, in a fixed order: post-NMS predictions that no GT box
    of the same class matches at IoU >= cfg.iou (the same matching the Detection Scores metric uses).

    `detection_pred` is one sample's "object detection" output (no batch axis). Each entry holds the
    box in input-image pixels (x0, x1, y0, y1), the class name, the confidence, the best IoU with any
    GT box and the fp_type. FP k of an image -- the engine's row `<image>_fp_<k>` -- is entry k.

    fp_type says why the prediction is false:
      "wrong_class" -- matched to a GT box of another class;
      "duplicate"   -- overlaps a GT box at the threshold that another prediction already took;
      "background"  -- overlaps no GT box at the threshold.
    The unlabeled split has no GT to judge against, so it yields none.
    """
    if preprocess.state == DataStateType.unlabeled:
        return []
    y_pred = detection_pred[None]
    s_prepro = SamplePreprocessResponse(np.array([sample_id]), preprocess)
    _, cls_gt, bbox_gt, predn = _prepare_detection_batch(y_pred, s_prepro)
    if len(predn) == 0:
        return []
    pred_gt_match, pred_is_tp, _ = _match_detections(predn, cls_gt, bbox_gt, cfg.iou)
    max_iou = (box_iou(predn[:, :4], bbox_gt).numpy().max(axis=1) if len(cls_gt)
               else np.zeros(len(predn), dtype=np.float32))
    # Same NMS output as predn, before scaling to the original image: boxes in model-input pixels,
    # the frame the input encoder's image (and so the GT instance masks) is in.
    pred = predictor.postprocess(torch.from_numpy(y_pred.copy()))[0].cpu().numpy()

    boxes = []
    for i in np.flatnonzero(~pred_is_tp):
        x0, y0, x1, y1 = pred[i, :4]
        # x0/y0 stop one pixel short of the edge, so a box at (or past) it still gets a 1-pixel mask.
        x0, x1 = int(np.clip(x0, 0, width - 1)), int(np.clip(x1, 0, width))
        y0, y1 = int(np.clip(y0, 0, height - 1)), int(np.clip(y1, 0, height))
        boxes.append({
            "x0": x0, "x1": max(x1, x0 + 1), "y0": y0, "y1": max(y1, y0 + 1),
            "class_name": all_clss.get(int(pred[i, 5]), "Unknown Class"),
            "confidence": float(pred[i, 4]),
            "max_iou": float(max_iou[i]),
            "fp_type": ("wrong_class" if pred_gt_match[i] >= 0
                        else "duplicate" if max_iou[i] >= cfg.iou else "background"),
        })
    return boxes


@tensorleap_false_positive_instances_encoder(
    "image", instance_metadata_types=FALSE_POSITIVE_METADATA_TYPES
)
def false_positive_instances(
    sample_id: str, preprocess: PreprocessResponse, predictions: Dict[str, np.ndarray]
) -> List[ElementInstance]:
    """One ElementInstance per false_positive_boxes entry: the box as a mask of ones."""
    if preprocess.state == DataStateType.unlabeled:
        return []  # no GT to judge against; skip loading the image
    image = input_encoder(sample_id, preprocess)
    height, width = image.shape[-2], image.shape[-1]
    instances = []
    for box in false_positive_boxes(sample_id, preprocess, predictions[DETECTION_PREDICTION],
                                    width, height):
        x0, x1, y0, y1 = box["x0"], box["x1"], box["y0"], box["y1"]
        mask = np.zeros(image.shape, dtype=np.float32)
        mask[..., y0:y1, x0:x1] = 1.0
        crop = image[..., y0:y1, x0:x1]
        w, h = (x1 - x0) / width, (y1 - y0) / height
        instances.append(ElementInstance(
            name=box["class_name"],
            mask=mask,
            instance_metadata={
                "instance_class": box["class_name"],
                "instance_bbox_area": float(w * h),
                "instance_bbox_aspect_ratio": float(w / h) if h > 0 else 0.0,
                "instance_center_x": float((x0 + x1) / 2 / width),
                "instance_center_y": float((y0 + y1) / 2 / height),
                "instance_crop_mean_brightness": float(crop.mean()),
                "instance_crop_std_brightness": float(crop.std()),
                "fp_confidence": box["confidence"],
                "fp_type": box["fp_type"],
            },
            instance_metrics={
                "instance_iou": box["max_iou"],
                "instance_confidence": box["confidence"],
            },
        ))
    return instances


@tensorleap_false_positive_custom_latent_space("instance_bbox_appearance_ls")
def fp_bbox_appearance_ls(
    sample_id: str, preprocess: PreprocessResponse, fp_instance: ElementInstance
) -> np.ndarray:
    """The GT instance LS (instance_bbox_appearance_ls) for a false-positive row: the box is read back
    from the instance mask's extent (the FP mask is a box of ones in input pixels)."""
    image = input_encoder(sample_id, preprocess)
    height, width = image.shape[-2], image.shape[-1]
    ys, xs = np.nonzero(np.asarray(fp_instance.mask).reshape(-1, height, width).max(axis=0) > 0)
    if xs.size == 0:
        return np.zeros(6, dtype=np.float32)  # empty mask: no box to describe
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
    w, h = (x1 - x0) / width, (y1 - y0) / height
    return _bbox_appearance_vector(
        image, (x0 + x1) / 2 / width, (y0 + y1) / 2 / height, w, h, x0, x1, y0, y1
    )


@tensorleap_false_positive_empty_gt("classes")
def false_positive_empty_gt(idx: str, preprocessing: PreprocessResponse) -> np.ndarray:
    """No object: the GT encoder's shape with every row padding (NaN), as for an image with no boxes."""
    return np.full((MAX_GT_BOXES, 5), np.nan, dtype=np.float32)
