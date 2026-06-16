import json
import os
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
import yaml
from code_loader.contract.datasetclasses import PreprocessResponse
from code_loader.contract.responsedataclasses import BoundingBox
from ultralytics.data import  build_yolo_dataset
from ultralytics.utils.plotting import output_to_target
from ultralytics.tensorleap_folder.global_params import cfg, all_clss, predictor


_CONFIG_PATH = Path(__file__).resolve().parent / 'tensorleap_config.yaml'
with open(_CONFIG_PATH) as _cfg_file:
    _TL_CONFIG = yaml.safe_load(_cfg_file)

MERGE_LABEL = _TL_CONFIG['merge_label']
EVAL_CLASS_MERGE = {int(k): int(v) for k, v in (_TL_CONFIG.get('eval_class_merge') or {}).items()}
FAMILY_TO_CLASS = dict(_TL_CONFIG['family_to_class'])

_AGG_MAP_PATH = Path(__file__).resolve().parents[2] / _TL_CONFIG['aggressor_map_filename']
try:
    with open(_AGG_MAP_PATH) as _agg_file:
        AGGRESSOR_MAP = json.load(_agg_file)
except FileNotFoundError:
    AGGRESSOR_MAP = {}

FALSE_AGGRESSOR_STEMS = {stem for stem, info in AGGRESSOR_MAP.items()
                         if info.get("role") == "false_aggressor"}

CLASS_NAME_TO_ID = {name: idx for idx, name in all_clss.items()}
FAMILY_TO_CLASS_ID = {fam: CLASS_NAME_TO_ID[cls] for fam, cls in FAMILY_TO_CLASS.items()
                      if cls in CLASS_NAME_TO_ID}


def _filtered_img_list(img_path, exclude_stems):
    # Drop excluded stems from a txt image-list (dir inputs pass through unchanged).
    # Use os.path.abspath, not resolve(): the images are symlinks and YOLO derives
    # labels by swapping '/images/'->'/labels/' in the image path, so the symlink
    # must stay in place or every image loads as a background (0 GT).
    if not exclude_stems or not str(img_path).endswith(".txt"):
        return img_path
    parent = Path(img_path).parent
    kept = []
    with open(img_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if Path(line).stem in exclude_stems:
                continue
            kept.append(os.path.abspath(parent / line[2:]) if line.startswith("./") else line)
    out = Path(tempfile.gettempdir()) / f"{Path(img_path).stem}.filtered_{len(kept)}.txt"
    out.write_text("\n".join(kept) + "\n")
    return str(out)


def create_data_with_ult(cfg,yolo_data, phase='val', exclude_stems=None):
    img_path = _filtered_img_list(yolo_data[phase], exclude_stems)
    n_samples = sum(1 for _ in open(img_path)) if str(img_path).endswith(".txt") else len(
        os.listdir(img_path))
    dataset = build_yolo_dataset(cfg, img_path,n_samples , yolo_data, mode='val', stride=32)
    # Report the real dataset length: the txt line count can differ (dropped images /
    # stale .cache) and would index past self.labels during the parse-time sweep.
    return dataset, len(dataset)

def pre_process_dataloader(preprocessresponse:PreprocessResponse, idx, predictor):
    idx = base_idx(idx)
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
    if Path(arch_name).stem not in  supported_versions +['None_path']:
        raise Exception(f"unsupported model. use one of {supported_versions} backbones")
    if (pt_name not in supported_versions and Path(arch_name).stem not in supported_versions +['None_path']) or (pt_name in supported_versions and arch_name!=pt_name and arch_name !='None_path') :
        raise Exception(f"unsupported model. use one of {supported_versions} backbones")

def get_dataset_split(phase, split_file):
    data = np.load(split_file)

    d = {
        "val": data["val_idxs"],
        "test": data["test_idxs"],
        "train": data["train_labeled_idxs"],
        "unlabeled": data["train_unlabeled_idxs"]
    }
    return [int(x) for x in d[phase]]

def set_leap_yaml2root(cfg):
    assert cfg.task == "detect", "Running detect leap binder while default.yaml task is set to pose"
    root = Path(__file__).resolve().parents[2]
    shutil.copy(Path(__file__).resolve().parent / 'detect'/'leap.yaml', root / 'leap.yaml')


def base_idx(idx):
    return int(str(idx).split('_')[0])


def instance_aggressor_role(stem, instance_cls_id):
    info = AGGRESSOR_MAP.get(stem)
    if not info:
        return "non_aggressor"
    role = info.get("role")
    if role == "non_aggressor":
        return "non_aggressor"
    target = FAMILY_TO_CLASS_ID.get(info.get("family"))
    is_target = target is not None and int(instance_cls_id) == target
    if role == "aggressor":
        return "aggressor" if is_target else "context"
    if role == "false_aggressor":
        return "false_aggressor" if is_target else "context"
    return "non_aggressor"


def merge_eval_cls(cls):
    if hasattr(cls, "clone"):
        out = cls.clone()
        for src, dst in EVAL_CLASS_MERGE.items():
            out[out == src] = dst
        return out
    out = np.asarray(cls).copy()
    for src, dst in EVAL_CLASS_MERGE.items():
        out[out == src] = dst
    return out


def eval_cls_name(idx):
    idx = int(idx)
    if idx in EVAL_CLASS_MERGE:
        idx = EVAL_CLASS_MERGE[idx]
    if idx in EVAL_CLASS_MERGE.values():
        return MERGE_LABEL
    return all_clss.get(idx, "Unknown Class")


def finite_or_none(d):
    # JSON / Elasticsearch cannot encode NaN or Infinity; index non-finite values as null.
    return {k: (None if isinstance(v, (float, np.floating)) and not np.isfinite(v) else v)
            for k, v in d.items()}


def instance_parts(preprocess):
    sid = str(np.asarray(preprocess.sample_ids).reshape(-1)[0])
    mapping = getattr(preprocess.preprocess_response, 'instance_to_sample_ids_mappings', None)
    base_id = mapping.get(sid, sid) if mapping else sid
    instance_idx = int(sid.rsplit('_', 1)[1]) if base_id != sid else None
    return base_id, instance_idx


def decoded_pred_boxes(image_chw, predictions):
    y_pred = predictor.postprocess(torch.from_numpy(predictions.copy()))
    _, cls_temp, bbx_temp, conf_temp = output_to_target(y_pred, max_det=predictor.args.max_det)
    t_pred = np.concatenate([bbx_temp, np.expand_dims(conf_temp, 1), np.expand_dims(cls_temp, 1)], axis=1)
    t_pred = t_pred[t_pred[:, 4] > (getattr(cfg, "conf", 0.25) or 0.25)]
    t_pred[:, :4:2] /= image_chw.shape[1]
    t_pred[:, 1:4:2] /= image_chw.shape[2]
    return t_pred


def to_labeled_bboxes(rows, label, conf=None):
    return [BoundingBox(x=float(r[0]), y=float(r[1]), width=float(r[2]), height=float(r[3]),
                        confidence=float(conf if conf is not None else r[4]), label=label)
            for r in rows]


def base_image_and_gt(preprocess, input_encoder, gt_encoder):
    base_id, inst = instance_parts(preprocess)
    img = input_encoder(base_id, preprocess.preprocess_response)
    img = img[0] if img.ndim == 4 else img
    gt = gt_encoder(base_id, preprocess.preprocess_response)
    gt = gt[0] if gt.ndim == 3 else gt
    gt = gt[~np.isnan(gt).any(axis=1)]
    return img, gt, inst


def resolve_instance_sample_id(preprocess):
    id_ = str(np.asarray(preprocess.sample_ids).reshape(-1)[0])
    mapping = getattr(preprocess.preprocess_response, 'instance_to_sample_ids_mappings', None)
    return mapping.get(id_, id_) if mapping else id_


def instance_pred_match_setup(y_pred, preprocess, instances_length_encoder):
    sample_id = resolve_instance_sample_id(preprocess)
    n_instances = instances_length_encoder(sample_id, preprocess.preprocess_response)
    if n_instances == 0:
        return n_instances, None, None, None
    batch = preprocess.preprocess_response.data['dataloader'][base_idx(sample_id)]
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
    return n_instances, gt_cls, gt_bbox, predn
