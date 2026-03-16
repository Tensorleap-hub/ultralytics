import os
from typing import List

import numpy as np
from code_loader.contract.datasetclasses import DataStateType, PreprocessResponse
from code_loader.inner_leap_binder.leapbinder_decorators import (
    tensorleap_gt_encoder,
    tensorleap_input_encoder,
    tensorleap_preprocess,
)

from ultralytics.tensorleap_folder.global_params import cfg, predictor, yolo_data
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


@tensorleap_preprocess()
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
                sample_ids=_limit_sample_ids(sample_ids),
                data={"dataloader": data_loader},
                sample_id_type=int,
                state=dataset_type,
            )
        )
    return responses


@tensorleap_input_encoder("image", channel_dim=1)
def input_encoder(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
    imgs, _, _, _ = pre_process_dataloader(preprocess, idx, predictor)
    return imgs.astype("float32")


@tensorleap_gt_encoder("classes")
def gt_encoder(idx: int, preprocessing: PreprocessResponse) -> np.ndarray:
    if preprocessing.state == DataStateType.unlabeled:
        return np.full((1, 5), np.nan, dtype=np.float32)
    _, clss, bboxes, _ = pre_process_dataloader(preprocessing, idx, predictor)
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
