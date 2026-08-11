import os
from code_loader.contract.datasetclasses import SamplePreprocessResponse
import onnxruntime as ort
import numpy as np
from ultralytics.tensorleap_folder.utils import validate_supported_models, set_leap_yaml2root
from ultralytics.tensorleap_folder.global_params import cfg, all_clss
from code_loader.plot_functions.visualize import visualize
from code_loader.contract.datasetclasses import PredictionTypeHandler
from code_loader.inner_leap_binder.leapbinder_decorators import tensorleap_load_model, tensorleap_integration_test
from leap_integration_lib import (
    bb_decoder,
    confusion_matrix_metric,
    cost,
    detection_scores,
    gt_bb_decoder,
    gt_encoder,
    image_visualizer,
    input_encoder,
    ious,
    loss,
    metadata_per_img,
    preprocess_func_leap,
)

prediction_type1 = PredictionTypeHandler(name='object detection', labels=["x", "y", "w", "h"] + [cl for cl in all_clss.values()], channel_dim=1)
prediction_type2 = PredictionTypeHandler(name='concatenate_20', labels=[str(i) for i in range(20)], channel_dim=-1)
prediction_type3 = PredictionTypeHandler(name='concatenate_40', labels=[str(i) for i in range(40)], channel_dim=-1)
prediction_type4 = PredictionTypeHandler(name='concatenate_80', labels=[str(i) for i in range(80)], channel_dim=-1)

@tensorleap_load_model([prediction_type1,prediction_type2,prediction_type3,prediction_type4])
def load_model():
    model_path="models/yolo11s.onnx"
    m_path = model_path if model_path != None else 'None_path'
    print("started custom tests")
    validate_supported_models(os.path.basename(cfg.model), m_path)
    if not os.path.exists(m_path):
        from export_model_to_tf import onnx_exporter  # TODO - currently supports only onnx
        m_path = onnx_exporter()
    model = ort.InferenceSession(m_path)
    return model


@tensorleap_integration_test()
def check_custom_test_mapping(idx, subset):
    s_prepro = SamplePreprocessResponse(np.array(idx), subset)
    image = input_encoder(idx, subset)
    model = load_model()
    y_pred = model.run(None, {'images': image})
    # get gt
    gt = gt_encoder(idx, subset)
    # custom metrics
    total_loss=loss(y_pred[1],y_pred[2],y_pred[3],gt ,y_pred[0])
    cost_dic=cost(y_pred[1],y_pred[2],y_pred[3],gt)
    # iou=ious(y_pred[0], s_prepro)
    # scores = detection_scores(y_pred[0], s_prepro)
    conf_mat = confusion_matrix_metric(y_pred[0], s_prepro)
    # metadata
    meta_data=metadata_per_img(idx, subset)
    # vis
    img_vis=image_visualizer(image)
    pred_img=bb_decoder(image,y_pred[0])
    gt_img = gt_bb_decoder(image, gt)
    visualize(img_vis)
    visualize(pred_img)
    visualize(gt_img)



if __name__ == '__main__':
    set_leap_yaml2root(cfg)
    check_custom_test_mapping('0', preprocess_func_leap()[1])
