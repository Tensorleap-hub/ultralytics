import os
from typing import List

import numpy as np
from code_loader.contract.datasetclasses import (
    DataStateType,
    ElementInstance,
    PreprocessResponse,
)
from code_loader.inner_leap_binder.leapbinder_decorators import (
    tensorleap_element_instance_preprocess,
    tensorleap_gt_encoder,
    tensorleap_input_encoder,
    tensorleap_instances_length_encoder,
    tensorleap_instances_masks_encoder,
)

from ultralytics.tensorleap_folder.global_params import all_clss, cfg, predictor, yolo_data
from ultralytics.tensorleap_folder.utils import (
    create_data_with_ult,
    get_dataset_split,
    pre_process_dataloader,
)


def _limit_sample_ids(sample_ids):
    max_samples = getattr(cfg, "max_samples", None)
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
    gt = gt_encoder(sample_id, preprocess)
    return gt[~np.isnan(gt).any(axis=1)]


@tensorleap_instances_length_encoder("image")
def instance_length_encoder(sample_id: str, preprocess: PreprocessResponse) -> int:
    return int(_valid_gt_boxes(sample_id, preprocess).shape[0])


@tensorleap_instances_masks_encoder("image")
def instance_mask_encoder(
    sample_id: str, preprocess: PreprocessResponse, instance_id: int
) -> ElementInstance:
    cx, cy, w, h, cls = _valid_gt_boxes(sample_id, preprocess)[instance_id]
    image = input_encoder(sample_id, preprocess)
    height, width = image.shape[-2], image.shape[-1]
    x0, x1 = int(np.clip((cx - w / 2) * width, 0, width)), int(np.clip((cx + w / 2) * width, 0, width))
    y0, y1 = int(np.clip((cy - h / 2) * height, 0, height)), int(np.clip((cy + h / 2) * height, 0, height))
    mask = np.zeros(image.shape, dtype=np.float32)
    mask[..., y0 : max(y1, y0 + 1), x0 : max(x1, x0 + 1)] = 1.0
    return ElementInstance(name=all_clss.get(int(cls), "Unknown Class"), mask=mask)


@tensorleap_element_instance_preprocess(instance_length_encoder, instance_mask_encoder)
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
                sample_ids=[str(i) for i in _limit_sample_ids(sample_ids)],
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


@tensorleap_gt_encoder("classes")
def gt_encoder(idx: str, preprocessing: PreprocessResponse) -> np.ndarray:
    if preprocessing.state == DataStateType.unlabeled:
        return np.full((1, 5), np.nan, dtype=np.float32)
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
