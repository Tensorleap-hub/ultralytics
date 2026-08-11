from typing import Dict, Union

import numpy as np
from code_loader.contract.datasetclasses import PreprocessResponse
from code_loader.inner_leap_binder.leapbinder_decorators import tensorleap_metadata

from leap_integration_lib.data import _sample_index, gt_encoder
from ultralytics.tensorleap_folder.global_params import (
    all_clss,
    possible_float_like_nan_types,
)
from ultralytics.tensorleap_folder.utils import (
    bbox_area_and_aspect_ratio,
    calculate_iou_all_pairs,
    update_dict_count_cls,
)


@tensorleap_metadata("metadata_sample_index")
def metadata_sample_index(idx: str, preprocess: PreprocessResponse) -> int:
    return _sample_index(idx)


@tensorleap_metadata("image info a", metadata_type=possible_float_like_nan_types)
def metadata_per_img(idx: str, data: PreprocessResponse) -> Dict[str, Union[str, int, float]]:
    nan_default_value = None
    img_idx = _sample_index(idx)
    gt_data = gt_encoder(idx, data)
    cls_gt = np.expand_dims(gt_data[:, 4], axis=1)
    bbox_gt = gt_data[:, :4]
    clss_info = np.unique(cls_gt, return_counts=True)
    count_dict = update_dict_count_cls(all_clss, clss_info, nan_default_value)
    areas, _ = bbox_area_and_aspect_ratio(
        bbox_gt, data.data["dataloader"][img_idx]["resized_shape"]
    )
    occlusion_matrix, areas_in_pixels, _ = calculate_iou_all_pairs(
        bbox_gt, data.data["dataloader"][img_idx]["resized_shape"]
    )
    no_nans_values = ~np.isnan(clss_info[0]).any()
    metadata = {
        "image path": data.data["dataloader"].im_files[img_idx],
        "idx": img_idx,
        "# unique classes": len(clss_info[0]) if no_nans_values else nan_default_value,
        "# of objects": int(clss_info[1].sum()) if no_nans_values else nan_default_value,
        "mean bbox area": float(areas.mean()) if no_nans_values else nan_default_value,
        "var bbox area": float(areas.var()) if no_nans_values else nan_default_value,
        "median bbox area": float(np.median(areas)) if no_nans_values else nan_default_value,
        "max bbox area": float(np.max(areas)) if no_nans_values else nan_default_value,
        "min bbox area": float(np.min(areas)) if no_nans_values else nan_default_value,
        "bbox overlap": float(occlusion_matrix.sum() / areas_in_pixels.sum())
        if no_nans_values
        else nan_default_value,
        "max bbox overlap": float((occlusion_matrix.sum(axis=1) / areas_in_pixels).max())
        if no_nans_values
        else nan_default_value,
    }
    metadata.update(**count_dict)
    return metadata
