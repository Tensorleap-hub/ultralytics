from code_loader.contract.datasetclasses import SamplePreprocessResponse
from code_loader.contract.enums import DataStateType
from code_loader import leap_binder
from code_loader.contract.datasetclasses import PreprocessResponse, DataStateType, SamplePreprocessResponse, \
    ConfusionMatrixElement
from code_loader.contract.enums import LeapDataType, MetricDirection, ConfusionMatrixValue
from code_loader.visualizers.default_visualizers import LeapImage
from code_loader.inner_leap_binder.leapbinder_decorators import (tensorleap_preprocess, tensorleap_gt_encoder,
                                                                 tensorleap_input_encoder, tensorleap_metadata,
                                                                 tensorleap_custom_visualizer)
from code_loader.contract.responsedataclasses import BoundingBox
from code_loader.contract.visualizer_classes import LeapImageWithBBox
from code_loader.utils import rescale_min_max
from code_loader.inner_leap_binder.leapbinder_decorators import tensorleap_custom_loss, tensorleap_custom_metric


import yaml
from types import SimpleNamespace
from code_loader.contract.enums import DatasetMetadataType
from coremltools.converters.mil.testing_reqs import tf

from ultralytics.utils import yaml_load
from ultralytics.utils.checks import check_file
import __main__, sys



import os
import re
import shutil
from pathlib import Path

import numpy as np
import torch
from code_loader.contract.datasetclasses import PreprocessResponse
from ultralytics.utils.plotting import output_to_target #change to our version








def create_mapping_and_test():
    if check_generic:
        leap_binder.check()
    m_path= model_path if model_path!=None else 'None_path'
    print("started custom tests")
    validate_supported_models(os.path.basename(cfg.model),m_path)
    if not os.path.exists(m_path):
        from export_model_to_tf import onnx_exporter #TODO - currently supports only onnx
        m_path=onnx_exporter()
        extract_mapping(m_path,mapping_version)
    keras_model=m_path.endswith(".h5")
    model = tf.keras.models.load_model(m_path) if keras_model else ort.InferenceSession(m_path)
    responses = preprocess_func_leap()
    for subset in responses: # [training, validation, test ,unlabeled]
        for idx in range(10):
            s_prepro=SamplePreprocessResponse(np.array(idx), subset)

            # get input images
            image = input_encoder(idx, subset)
            concat = np.expand_dims(image, axis=0)

            # predict
            y_pred = model([concat]) if keras_model else model.run(None, {model.get_inputs()[0].name: concat})
            if not keras_model:
                y_pred=[tf.convert_to_tensor(p)  for p in y_pred]
            if subset.state != DataStateType.unlabeled:

                # get gt
                gt = gt_encoder(idx, subset)
                gt_img = gt_bb_decoder(np.expand_dims(image, axis=0), np.expand_dims(gt, axis=0))

                # custom metrics
                total_loss=loss(y_pred[1].numpy(),y_pred[2].numpy(),y_pred[3].numpy(),np.expand_dims(gt,axis=0), y_pred[0].numpy())
                cost_dic=cost(y_pred[1].numpy(),y_pred[2].numpy(),y_pred[3].numpy(),np.expand_dims(gt,axis=0))
                iou=ious(y_pred[0].numpy(), s_prepro)
                conf_mat = confusion_matrix_metric(y_pred[0].numpy(), s_prepro)

            # metadata
            meta_data=metadata_per_img(idx, subset)

            # vis
            img_vis=image_visualizer(np.expand_dims(image,axis=0))
            pred_img=bb_decoder(np.expand_dims(image,axis=0),y_pred[0].numpy())
            if plot_vis:
                visualize(img_vis)
                visualize(pred_img)
                if subset.state != DataStateType.unlabeled:
                    visualize(gt_img)
    print("finish tests")



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




def get_dataset_yaml(cfg):
    dataset_yaml_file=check_file(cfg.data)
    return  yaml_load(dataset_yaml_file, append_filename=True)



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
    root = Path(__file__).resolve().parent.parent
    file_path = os.path.join(root, 'cfg/tl_default.yaml') if not dir_path else dir_path
    with open(file_path, 'r') as file:
        config_dict = yaml.safe_load(file)
    if isinstance(config_dict, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in config_dict.items()})
    elif isinstance(config_dict, list):
        return [dict_to_namespace(i) for i in config_dict]
    else:
        return config_dict

def set_global_params():
    test_pr=detect_entry_script()
    dir_path= get_entry_var("DIR_PATH") if test_pr else False
    cfg = set_cfg_dict(dir_path)
    dataset_yaml=get_dataset_yaml(cfg)#doable
    all_clss=dataset_yaml["names"]
    cls_mapping = {v: k for k, v in all_clss.items()}
    wanted_cls_dic=get_wanted_cls(cls_mapping,cfg)
    possible_float_like_nan_types={f"count of '{v}' class ({k})": DatasetMetadataType.float   for k, v in all_clss.items()}
    return test_pr,cfg,all_clss,wanted_cls_dic,possible_float_like_nan_types




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




if __name__ == '__main__':
    check_generic = True
    plot_vis= False
    model_path = None  # Choose None if only pt version available else, use your h5/onnx model's path.
    mapping_version = None # Set as  None if the model's name is supported by ultralytics. Else, set to the base yolo architecture name (e.x if your trained model has the same architecture as yolov11s set mapping_version=yolov11s ) .
    # check_custom_test()
