import os
from typing import List

import numpy as np
from code_loader.contract.datasetclasses import (
    DataStateType,
    ElementInstance,
    PreprocessResponse,
)
from code_loader.contract.enums import DatasetMetadataType
from code_loader.inner_leap_binder.leapbinder_decorators import (
    tensorleap_custom_latent_space,
    tensorleap_element_instance_preprocess,
    tensorleap_gt_encoder,
    tensorleap_input_encoder,
    tensorleap_instance_custom_latent_space,
    tensorleap_instances_length_encoder,
    tensorleap_instances_masks_encoder,
)

from ultralytics.tensorleap_folder.global_params import all_clss, cfg, predictor, yolo_data
from ultralytics.tensorleap_folder.utils import (
    create_data_with_ult,
    get_dataset_split,
    pre_process_dataloader,
)


def _limit_sample_ids(sample_ids, phase):
    """The first `max_samples` ids of a phase. `skip_samples_<phase>` drops ids from its start and
    `max_samples_<phase>` overrides the cap for that phase alone, so phases that read the same image
    folder (train / unlabeled / additional on train2017) can take disjoint windows of it."""
    skip = int(getattr(cfg, f"skip_samples_{phase}", 0) or 0)
    max_samples = getattr(cfg, f"max_samples_{phase}", None) or getattr(cfg, "max_samples", None)
    sample_ids = sample_ids[skip:]
    if max_samples is None:
        return sample_ids
    return sample_ids[: int(max_samples)]


def _sample_index(sample_id) -> int:
    """Element-instance ids arrive as '<sample_id>_<instance_id>'; keep only the image index.

    Metrics pass SamplePreprocessResponse.sample_ids, which is a 1-element sequence rather than a
    scalar, so unwrap it before parsing.
    """
    if isinstance(sample_id, (list, tuple, np.ndarray)):
        sample_id = np.asarray(sample_id).reshape(-1)[0]
    return int(str(sample_id).split("_")[0])


def _valid_gt_boxes(sample_id, preprocess: PreprocessResponse) -> np.ndarray:
    """The image's instances: its complete annotation rows, in GT order (capped like the GT), in
    every split -- the unlabeled one included, whose ground truth is empty."""
    rows = _annotation_rows(sample_id, preprocess)[:MAX_GT_BOXES]
    return rows[~np.isnan(rows).any(axis=1)]


@tensorleap_instances_length_encoder("image")
def instance_length_encoder(sample_id: str, preprocess: PreprocessResponse) -> int:
    return int(_valid_gt_boxes(sample_id, preprocess).shape[0])


def _bbox_pixel_bounds(image: np.ndarray, cx: float, cy: float, w: float, h: float):
    height, width = image.shape[-2], image.shape[-1]
    x0, x1 = int(np.clip((cx - w / 2) * width, 0, width)), int(np.clip((cx + w / 2) * width, 0, width))
    y0, y1 = int(np.clip((cy - h / 2) * height, 0, height)), int(np.clip((cy + h / 2) * height, 0, height))
    return x0, max(x1, x0 + 1), y0, max(y1, y0 + 1)


def _bbox_appearance_vector(image: np.ndarray, cx: float, cy: float, w: float, h: float,
                            x0: int, x1: int, y0: int, y1: int) -> np.ndarray:
    """The instance LS vector [cx, cy, w, h (normalized), crop mean, crop std] for the pixel crop."""
    crop = image[..., y0:y1, x0:x1]
    return np.array([cx, cy, w, h, float(crop.mean()), float(crop.std())], dtype=np.float32)


@tensorleap_instances_masks_encoder("image")
def instance_mask_encoder(
    sample_id: str, preprocess: PreprocessResponse, instance_id: int
) -> ElementInstance:
    cx, cy, w, h, cls = _valid_gt_boxes(sample_id, preprocess)[instance_id]
    image = input_encoder(sample_id, preprocess)
    x0, x1, y0, y1 = _bbox_pixel_bounds(image, cx, cy, w, h)
    mask = np.zeros(image.shape, dtype=np.float32)
    mask[..., y0:y1, x0:x1] = 1.0
    crop = image[..., y0:y1, x0:x1]
    class_name = all_clss.get(int(cls), "Unknown Class")
    # Keep this key set identical across instances; the types are declared via
    # instance_metadata_types on the preprocess decorator (INSTANCE_METADATA_TYPES).
    return ElementInstance(
        name=class_name,
        mask=mask,
        instance_metadata={
            "instance_class": class_name,
            "instance_bbox_area": float(w * h),
            "instance_bbox_aspect_ratio": float(w / h) if h > 0 else 0.0,
            "instance_center_x": float(cx),
            "instance_center_y": float(cy),
            "instance_crop_mean_brightness": float(crop.mean()),
            "instance_crop_std_brightness": float(crop.std()),
        },
    )


@tensorleap_instance_custom_latent_space(name="instance_bbox_appearance_ls")
def instance_bbox_appearance_ls(
    sample_id: str, preprocess: PreprocessResponse, instance_id: int
) -> np.ndarray:
    cx, cy, w, h, cls = _valid_gt_boxes(sample_id, preprocess)[instance_id]
    image = input_encoder(sample_id, preprocess)
    x0, x1, y0, y1 = _bbox_pixel_bounds(image, cx, cy, w, h)
    return _bbox_appearance_vector(image, cx, cy, w, h, x0, x1, y0, y1)


@tensorleap_custom_latent_space(name="image_appearance_ls")
def image_appearance_ls(sample_id: str, preprocess: PreprocessResponse) -> np.ndarray:
    """Image-level custom LS, next to the per-instance one above: the image's per-channel
    brightness mean and std, its GT box count and mean box area (normalized)."""
    image = input_encoder(sample_id, preprocess)
    boxes = _valid_gt_boxes(sample_id, preprocess)
    mean_area = float((boxes[:, 2] * boxes[:, 3]).mean()) if len(boxes) else 0.0
    return np.concatenate([image.mean(axis=(-2, -1)), image.std(axis=(-2, -1)),
                           [float(len(boxes)), mean_area]]).astype(np.float32)


# Declaring the types (code-loader >= 1.0.206) makes the columns independent of
# probing: a key stays typed even when the first probed instance lacks a value.
INSTANCE_METADATA_TYPES = {
    "instance_class": DatasetMetadataType.string,
    "instance_bbox_area": DatasetMetadataType.float,
    "instance_bbox_aspect_ratio": DatasetMetadataType.float,
    "instance_center_x": DatasetMetadataType.float,
    "instance_center_y": DatasetMetadataType.float,
    "instance_crop_mean_brightness": DatasetMetadataType.float,
    "instance_crop_std_brightness": DatasetMetadataType.float,
}


@tensorleap_element_instance_preprocess(
    instance_length_encoder,
    instance_mask_encoder,
    instance_metadata_types=INSTANCE_METADATA_TYPES,
)
def preprocess_func_leap() -> List[PreprocessResponse]:
    dataset_types = [DataStateType.training, DataStateType.validation]
    phases = ["train", "val"]
    responses = []
    if cfg.tensorleap_use_test:
        phases.append("test")
        dataset_types.append(DataStateType.test)
    if cfg.tensorleap_use_unlabeled:
        phases.append("unlabeled")
        dataset_types.append(DataStateType.unlabeled)
    if getattr(cfg, "tensorleap_use_additional", False):
        phases.append("additional")
        dataset_types.append(DataStateType.additional)
    for phase, dataset_type in zip(phases, dataset_types):
        data_loader, n_samples = create_data_with_ult(cfg, yolo_data, phase=phase)
        sample_ids = (
            list(range(n_samples))
            if not cfg.use_data_split_file[0]
            else get_dataset_split(
                phase,
                os.path.join(cfg.tensorleap_path, cfg.use_data_split_file[1]),
            )
        )
        responses.append(
            PreprocessResponse(
                sample_ids=[str(i) for i in _limit_sample_ids(sample_ids, phase)],
                data={"dataloader": data_loader},
                sample_id_type=str,
                state=dataset_type,
            )
        )
    return responses


@tensorleap_input_encoder("image", channel_dim=1)
def input_encoder(idx: str, preprocess: PreprocessResponse) -> np.ndarray:
    imgs, _, _, _ = pre_process_dataloader(preprocess, _sample_index(idx), predictor)
    return imgs.astype("float32")


def _gt_rows(idx: str, preprocessing: PreprocessResponse) -> np.ndarray:
    """The ground truth: the annotation rows, none (a single NaN row) for the unlabeled split."""
    if preprocessing.state == DataStateType.unlabeled:
        return np.full((1, 5), np.nan, dtype=np.float32)
    return _annotation_rows(idx, preprocessing)


def _annotation_rows(idx: str, preprocessing: PreprocessResponse) -> np.ndarray:
    """One row per annotated box, [cx, cy, w, h, cls], in every split; a single NaN row when the
    image has none. The instances are these boxes, so the unlabeled split has instances too (its
    images keep their label files here) while its ground truth stays empty."""
    _, clss, bboxes, _ = pre_process_dataloader(preprocessing, _sample_index(idx), predictor)
    if clss.shape[0] == 0 and bboxes.shape[0] == 0:
        return np.full((1, 5), np.nan, dtype=np.float32)
    if clss.shape[0] == 0:
        temp_array = np.full((bboxes.shape[0], 5), np.nan, dtype=np.float32)
        temp_array[:, :4] = bboxes
        return temp_array
    if bboxes.shape[0] == 0:
        temp_array = np.full((clss.shape[0], 5), np.nan, dtype=np.float32)
        temp_array[:, 4] = clss
        return temp_array
    return np.concatenate([bboxes, clss], axis=1)


# COCO's busiest image has 90 boxes. A fixed GT shape lets the engine batch samples together.
MAX_GT_BOXES = 100


@tensorleap_gt_encoder("classes")
def gt_encoder(idx: str, preprocessing: PreprocessResponse) -> np.ndarray:
    """(MAX_GT_BOXES, 5); rows past the image's boxes are NaN."""
    rows = _gt_rows(idx, preprocessing)[:MAX_GT_BOXES]
    padded = np.full((MAX_GT_BOXES, 5), np.nan, dtype=np.float32)
    padded[:len(rows)] = rows
    return padded
