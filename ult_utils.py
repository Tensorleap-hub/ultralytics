
from code_loader.contract.enums import DatasetMetadataType
from ultralytics.utils import callbacks as callbacks_ult
from ultralytics.models.yolo.detect import DetectionValidator #problematic
from ultralytics.utils import yaml_load
from ultralytics.utils.checks import check_file
from ultralytics import YOLO
from ultralytics.utils import IterableSimpleNamespace  # this what makes it problematic
from ultralytics.data import  build_yolo_dataset#problemtic
from ultralytics.utils.plotting import output_to_target #doable


import __main__, sys, os
import os
import shutil
import numpy as np
import torch
from pathlib import Path
from types import SimpleNamespace
import yaml
import re


from code_loader.contract.datasetclasses import PreprocessResponse





def yolo_version_check(model_path):
    model_name = model_path.stem
    match = re.fullmatch(r'yolov5([a-z])', model_name)
    exported_path= model_path.with_name(model_name + 'u' + '.onnx') if match else model_path.with_suffix('.onnx')
    if not exported_path.exists():
        raise FileNotFoundError(f"File {exported_path} not found, check the name of the yolo exported version.")
    return exported_path


def dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(i) for i in d]
    else:
        return d

def export_to_onnx(cfg):
    model = YOLO(cfg.model if hasattr(cfg, "model") else "yolo11s.pt")
    model.export(format="onnx", nms=False, export_train_head=True)

def onnx_exporter(cfg,h5=False):
    model_path=Path(cfg.model)
    if not model_path.is_absolute():
        model_path = (cfg.tensorleap_path / model_path).resolve()
    cfg.model=model_path
    export_to_onnx(cfg)
    exported_model_path=yolo_version_check(model_path)
    print(f"Model exported to ONNX: {exported_model_path}")
    return str(exported_model_path)

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



def detect_entry_script():
    loader = getattr(__main__, "__loader__", None)
    if loader and hasattr(loader, "archive"):
        return os.path.basename(loader.archive)=='tests_leap_custom_test.py'
    if hasattr(__main__, "__file__"):
        return os.path.basename(__main__.__file__)=='tests_leap_custom_test.py'
    return os.path.basename(sys.argv[0])=='tests_leap_custom_test.py'

def get_entry_var(var_name):
    return os.getenv(var_name)

def dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(i) for i in d]
    else:
        return d

def get_yolo_data(cfg):
    from ultralytics.data.utils import check_det_dataset
    return check_det_dataset(cfg.data, autodownload=True)

def get_criterion(model_path,cfg,test_pr):
    if not model_path.is_absolute():
        model_path = (cfg.tensorleap_path / model_path).resolve()
    # if not test_pr:
    #     assert model_path.is_relative_to(cfg.tensorleap_path), (
    #         f"❌ {model_path!r} is not inside tensorleap path {cfg.tensorleap_path!r}" )
    model_base = YOLO(model_path)
    criterion = model_base.init_criterion()
    criterion.hyp = IterableSimpleNamespace(**criterion.hyp)
    criterion.hyp.box = cfg.box
    criterion.hyp.cls = cfg.cls
    criterion.hyp.dfl = cfg.dfl
    return criterion

def get_dataset_yaml(cfg):
    dataset_yaml_file=check_file(cfg.data)
    return  yaml_load(dataset_yaml_file, append_filename=True)


def get_predictor_obj(cfg,yolo_data):
    callbacks = callbacks_ult.get_default_callbacks()
    predictor = DetectionValidator(args=cfg, _callbacks=callbacks)
    predictor.data = yolo_data
    predictor.end2end = False
    return predictor

def get_wanted_cls(cls_mapping,cfg):
    wanted_cls = cfg.wanted_cls
    supported_cls=np.isin(wanted_cls,list(cls_mapping.keys()))
    if not supported_cls.all():
        print(f"{list(np.array(wanted_cls)[~supported_cls])} objects are not supported and will not be shown in calculations.")
    wanted_cls =  np.array(wanted_cls)[supported_cls]
    if wanted_cls is None or len(wanted_cls)==0:
        wanted_cls = np.array(list(cls_mapping.keys())[:10])
        print(f"No wanted classes found, use the default top 10: {wanted_cls}")
    wanted_cls_dic = {k: cls_mapping[k] for k in wanted_cls}
    return wanted_cls_dic

def set_cfg_dict(dir_path=False):
    root = Path(__file__).resolve().parent
    file_path = os.path.join(root, 'tl_default.yaml') if not dir_path else dir_path
    with open(file_path, 'r') as file:
        config_dict = yaml.safe_load(file)
    if isinstance(config_dict, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in config_dict.items()})
    elif isinstance(config_dict, list):
        return [dict_to_namespace(i) for i in config_dict]
    else:
        return config_dict

def get_global_params():
    test_pr=detect_entry_script()
    dir_path= get_entry_var("DIR_PATH") if test_pr else False
    cfg = set_cfg_dict(dir_path)
    yolo_data=get_yolo_data(cfg) #doable
    dataset_yaml=get_dataset_yaml(cfg)#doable
    criterion=get_criterion(Path(cfg.model),cfg,test_pr)#problemtic
    all_clss=dataset_yaml["names"]
    cls_mapping = {v: k for k, v in all_clss.items()}
    wanted_cls_dic=get_wanted_cls(cls_mapping,cfg)
    predictor=get_predictor_obj(cfg,yolo_data)#problemtic
    possible_float_like_nan_types = {f"count of '{v}' class ({k})": DatasetMetadataType.float for k, v in
                                     all_clss.items()}
    return  dir_path, cfg, yolo_data, dataset_yaml, criterion, all_clss, cls_mapping, wanted_cls_dic, predictor, possible_float_like_nan_types