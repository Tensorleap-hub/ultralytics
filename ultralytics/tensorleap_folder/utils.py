import os
import re
import shutil
from pathlib import Path

import numpy as np
import torch
from code_loader.contract.datasetclasses import PreprocessResponse
from ultralytics.data import  build_yolo_dataset#problemtic
from ultralytics.utils.plotting import output_to_target #doable


def create_data_with_ult(cfg,yolo_data, phase='val'):
    n_samples = len(os.listdir(yolo_data[phase]))
    dataset = build_yolo_dataset(cfg, yolo_data[phase],n_samples , yolo_data, mode='val', stride=32)
    return dataset, n_samples

def pre_process_dataloader(preprocessresponse:PreprocessResponse, idx, predictor):
    batch= preprocessresponse.data['dataloader'][idx]
    batch = predictor.preprocess(batch)
    imgs, clss, bboxes, batch_idxs, ori_shape, resized_shape,ratio_pad = batch['img'], batch['cls'], batch['bboxes'], batch['batch_idx'],batch['ori_shape'],batch['resized_shape'],batch['ratio_pad']
    return imgs.numpy(), clss.numpy(), bboxes.numpy(), batch_idxs.numpy()


def pred_post_process(y_pred, predictor, image, cfg):
    y_pred = predictor.postprocess(torch.from_numpy(y_pred).unsqueeze(0))
    _, cls_temp, bbx_temp, conf_temp = output_to_target(y_pred, max_det=predictor.args.max_det)
    t_pred = np.concatenate([bbx_temp, np.expand_dims(conf_temp, 1), np.expand_dims(cls_temp, 1)], axis=1)
    post_proc_pred = t_pred[t_pred[:, 4] > (getattr(cfg, "conf", 0.3) or 0.3)]
    post_proc_pred[:, :4:2] /= image.shape[1]
    post_proc_pred[:, 1:4:2] /= image.shape[2]
    return post_proc_pred

def update_dict_count_cls(all_clss,clss_info,nan_default_value):
    if np.isnan(clss_info[0]).any():
        return {f"count of '{v}' class ({k})": nan_default_value   for k, v in all_clss.items()}
    return {f"count of '{v}' class ({k})": int(clss_info[1][clss_info[0]==k]) if k in clss_info[0] else nan_default_value for k, v in all_clss.items()}

def update_dict_bbox_cls_info(all_clss,info,clss_info,func_type='mean',task='area',nan_default_value=None):
    def get_mask(clss_info,k,info):
        mask=clss_info[:, 0] == k
        if info.ndim==2:
            mask=mask[:,None]*mask[None,:]
        return mask

    if np.isnan(info).any():
        return {f"{task}: {func_type} bbox of '{v}' class ({k})": nan_default_value   for k, v in all_clss.items()}
    if func_type=='mean':
        func=np.mean
    elif func_type=='var':
        func=np.var
    elif func_type=='min':
        func=np.min
    elif func_type=='max':
        func=np.max
    elif func_type=='diff':
        func = lambda x: np.max(x) - np.min(x)

    return {f"{task}: {func_type} bbox of '{v}' class ({k})": float(func(info[get_mask(clss_info,k,info)])) if k in clss_info else 0. for k, v in all_clss.items()}



def bbox_area_and_aspect_ratio(bboxes: np.ndarray, resized_shape):
    widths = bboxes[:, 2]
    heights = bboxes[:, 3]
    areas = widths * heights
    aspect_ratios = (heights*resized_shape[0]) / (widths*resized_shape[1])
    return areas, aspect_ratios




def calculate_iou_all_pairs(bboxes: np.ndarray, image_size: tuple):

    areas_in_pixels = (bboxes[:,2]*image_size[0]* bboxes[:,3]*image_size[1]).astype(np.float32)

    bboxes = np.asarray([xywh_to_xyxy_format(bbox[:-1]) for bbox in bboxes])
    bboxes[:,::2] *= image_size[0]
    bboxes[:,1::2] *= image_size[1]

    num_bboxes = len(bboxes)
    x_min = np.maximum(bboxes[:, 0][:, np.newaxis], bboxes[:, 0])
    y_min = np.maximum(bboxes[:, 1][:, np.newaxis], bboxes[:, 1])
    x_max = np.minimum(bboxes[:, 2][:, np.newaxis], bboxes[:, 2])
    y_max = np.minimum(bboxes[:, 3][:, np.newaxis], bboxes[:, 3])
    inter_w = np.clip(x_max - x_min, 0, None)
    inter_h = np.clip(y_max - y_min, 0, None)
    inter_area = inter_w * inter_h
    np.fill_diagonal(inter_area, 0)
    upper_tri_mask = np.triu(np.ones((num_bboxes, num_bboxes), dtype=bool), k=1)
    occlusion_matrix = inter_area * upper_tri_mask
    union_in_pixels= areas_in_pixels - np.sum(occlusion_matrix.T, axis=1)
    return occlusion_matrix.astype(np.float32), areas_in_pixels.astype(np.float32), union_in_pixels.astype(np.float32)

def xywh_to_xyxy_format(boxes):
    min_xy = boxes[..., :2] - boxes[..., 2:] / 2
    max_xy = boxes[..., :2] + boxes[..., 2:] / 2
    result = np.concatenate([min_xy, max_xy], -1)
    return result.astype(np.float32)

def extract_mapping(m_path,mapping_version):
    def extract_yolo_variant(filename):
        pattern = r'yolo(?:v)?\d+[a-zA-Z]'
        match = re.search(pattern, filename)
        if not match:
            return False
        else:
            return f"{match.group()}".replace('v','')

    filename=Path(m_path).stem if mapping_version==None else mapping_version
    model_type=extract_yolo_variant(filename)
    root = Path.cwd()
    mapping_folder_path =root / Path('ultralytics/tensorleap_folder/mapping')
    source_file = mapping_folder_path / f'leap_mapping_{model_type}.yaml'

    if not model_type or not os.path.exists(source_file):
        print(f"No Mapping for {m_path} was found, put your mapping in the root directory and check if it is supported.")
    else:
        destination_file = root/ 'leap_mapping.yaml'
        shutil.copy(source_file, destination_file)
        print(f"Extracting mapping for {model_type} completed")

def validate_supported_models(pt_name,arch_name):
    supported_versions = [
        "yolov5mu", "yolov5nu", "yolov5su",
        "yolov8n", "yolov8x",
        "yolov9c", "yolov9m", "yolov9s", "yolov9t",
        "yolo11x","yolo11m", "yolo11n", "yolo11s",
        "yolo12l", "yolo12m", "yolo12n", "yolo12s"
    ]
    if arch_name not in  supported_versions +['None_path']:
        raise Exception(f"unsupported model. use one of {supported_versions} backbones")
    if (pt_name not in supported_versions and arch_name not in supported_versions +['None_path']) or (pt_name in supported_versions and arch_name!=pt_name and arch_name !='None_path') :
        raise Exception(f"unsupported model. use one of {supported_versions} backbones")

