import copy
from pathlib import Path
import os
import numpy as np
import yaml
from types import SimpleNamespace
from code_loader.contract.enums import DatasetMetadataType

from ultralytics import YOLO
from ultralytics.models.yolo.pose import PosePredictor, PoseValidator
from ultralytics.utils import callbacks as callbacks_ult
from ultralytics.models.yolo.detect import DetectionValidator #problematic
from ultralytics.utils import yaml_load
from ultralytics.utils.checks import check_file


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

def get_criterion(model_path,cfg):
    from ultralytics import YOLO
    from ultralytics.utils import IterableSimpleNamespace #this what makes it problematic
    if not model_path.is_absolute():
        model_path = (cfg.tensorleap_path / model_path).resolve()
    assert model_path.is_relative_to(cfg.tensorleap_path), (
        f"❌ {model_path!r} is not inside tensorleap path {cfg.tensorleap_path!r}" )
    model_base = YOLO(model_path)
    criterion = model_base.init_criterion()
    criterion.hyp = IterableSimpleNamespace(**criterion.hyp)
    criterion.hyp.box = cfg.box
    criterion.hyp.cls = cfg.cls
    criterion.hyp.dfl = cfg.dfl
    criterion.hyp.pose = cfg.pose
    criterion.hyp.kobj = cfg.kobj

    return criterion

def get_dataset_yaml(cfg):
    dataset_yaml_file=check_file(cfg.data)
    return  yaml_load(dataset_yaml_file, append_filename=True)


def get_predictor_obj(cfg,yolo_data):
    callbacks = callbacks_ult.get_default_callbacks()
    predictor = PoseValidator(args=cfg, _callbacks=callbacks)
    # predictor.nc = 1
    predictor.data = yolo_data
    predictor.end2end = False
    model = YOLO(cfg.model if hasattr(cfg, "model") else "yolo11s.pt")
    predictor.init_metrics(model)
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

def set_cfg_dict():
    root = Path(__file__).resolve().parent.parent
    file_path = os.path.join(root, 'cfg/default.yaml')
    with open(file_path, 'r') as file:
        config_dict = yaml.safe_load(file)
    if isinstance(config_dict, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in config_dict.items()})
    elif isinstance(config_dict, list):
        return [dict_to_namespace(i) for i in config_dict]
    else:
        return config_dict

cfg = set_cfg_dict()
yolo_data=get_yolo_data(cfg) #doable
dataset_yaml=get_dataset_yaml(cfg)#doable
ob_cfg = copy.deepcopy(cfg)
ob_cfg.data = 'coco.yaml'
ob_cfg.task = 'detect'
ob_data_yaml=get_dataset_yaml(ob_cfg)
ob_yolo_data=get_yolo_data(ob_cfg)
criterion=get_criterion(Path(cfg.model),cfg)#problemtic
all_clss=dataset_yaml["names"]
ob_all_clss=ob_yolo_data["names"]
cls_mapping = {v: k for k, v in all_clss.items()}
wanted_cls_dic=get_wanted_cls(cls_mapping,cfg)
predictor=get_predictor_obj(cfg,yolo_data)#problemtic
possible_float_like_nan_types={f"count of '{v}' class ({k})": DatasetMetadataType.float   for k, v in all_clss.items()}