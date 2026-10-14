import numpy as np
import torch
from code_loader.contract.datasetclasses import SamplePreprocessResponse
from code_loader.contract.enums import LeapDataType
from code_loader.contract.responsedataclasses import BoundingBox
from code_loader.contract.visualizer_classes import LeapImageWithBBox
from code_loader.inner_leap_binder.leapbinder_decorators import tensorleap_custom_visualizer
from code_loader.utils import rescale_min_max
from code_loader.visualizers.default_visualizers import LeapImage
from ultralytics.utils.plotting import output_to_target

from leap_integration_lib.data import _valid_gt_boxes
from ultralytics.tensorleap_folder.global_params import all_clss, cfg, predictor


@tensorleap_custom_visualizer("bb_gt_decoder", LeapDataType.ImageWithBBox)
def gt_bb_decoder(image: np.ndarray, bb_gt: np.ndarray) -> LeapImageWithBBox:
    bbox = [
        BoundingBox(
            x=bbx[0],
            y=bbx[1],
            width=bbx[2],
            height=bbx[3],
            confidence=1,
            label=all_clss.get(int(bbx[4]) if not np.isnan(bbx[4]) else -1, "Unknown Class"),
        )
        for bbx in bb_gt.squeeze(0)
        if not np.isnan(bbx).any()  # padding rows
    ]
    image = rescale_min_max(image.squeeze(0))
    return LeapImageWithBBox(data=image.transpose(1, 2, 0), bounding_boxes=bbox)


@tensorleap_custom_visualizer("image_visualizer", LeapDataType.Image)
def image_visualizer(image: np.ndarray) -> LeapImage:
    image = rescale_min_max(image.squeeze(0))
    return LeapImage(image.transpose(1, 2, 0), compress=False)


@tensorleap_custom_visualizer("bb_decoder", LeapDataType.ImageWithBBox)
def bb_decoder(image: np.ndarray, predictions: np.ndarray) -> LeapImageWithBBox:
    image = image.squeeze(0)
    y_pred = predictor.postprocess(torch.from_numpy(predictions.copy()))
    _, cls_temp, bbx_temp, conf_temp = output_to_target(y_pred, max_det=predictor.args.max_det)
    t_pred = np.concatenate(
        [bbx_temp, np.expand_dims(conf_temp, 1), np.expand_dims(cls_temp, 1)],
        axis=1,
    )
    post_proc_pred = t_pred[t_pred[:, 4] > (getattr(cfg, "conf", 0.25) or 0.25)]
    post_proc_pred[:, :4:2] /= image.shape[1]
    post_proc_pred[:, 1:4:2] /= image.shape[2]
    bbox = [
        BoundingBox(
            x=bbx[0],
            y=bbx[1],
            width=bbx[2],
            height=bbx[3],
            confidence=bbx[4],
            label=all_clss.get(int(bbx[5]), "Unknown Class"),
        )
        for bbx in post_proc_pred
    ]
    image = rescale_min_max(image)
    return LeapImageWithBBox(data=image.transpose(1, 2, 0), bounding_boxes=bbox)


@tensorleap_custom_visualizer("instance_box", LeapDataType.ImageWithBBox)
def instance_box(image: np.ndarray, preprocess: SamplePreprocessResponse) -> LeapImageWithBBox:
    """An instance row: the (parent) frame with only this instance's GT box. An image row: all its GT boxes.

    The row is resolved through code-loader's instance mappings, not by parsing its id, and the
    instance's position follows the same `_valid_gt_boxes` order as the instance mask encoder.
    """
    sample_id = str(np.atleast_1d(preprocess.sample_ids)[0])
    response = preprocess.preprocess_response
    parent = (response.instance_to_sample_ids_mappings or {}).get(sample_id)
    if parent is None:
        boxes = _valid_gt_boxes(sample_id, response)
    else:
        position = [str(i) for i in response.sample_ids_to_instance_mappings[parent]].index(sample_id)
        boxes = _valid_gt_boxes(parent, response)[position:position + 1]
    bbox = [
        BoundingBox(
            x=bbx[0], y=bbx[1], width=bbx[2], height=bbx[3], confidence=1,
            label=all_clss.get(int(bbx[4]), "Unknown Class"),
        )
        for bbx in boxes
    ]
    image = rescale_min_max(image.squeeze(0))
    return LeapImageWithBBox(data=image.transpose(1, 2, 0), bounding_boxes=bbox)
