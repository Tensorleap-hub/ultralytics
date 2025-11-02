import os
from pathlib import Path
from types import SimpleNamespace
import yaml


def dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(i) for i in d]
    else:
        return d

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
