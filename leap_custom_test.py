import os
from code_loader.contract.datasetclasses import SamplePreprocessResponse
from code_loader.contract.enums import DataStateType

from leap_binder import (input_encoder, preprocess_func_leap, gt_encoder,
                         loss, gt_bb_decoder, image_visualizer, bb_decoder,
                         cost, metadata_per_img, ious, confusion_matrix_metric, preprocess_unlabeled_func_leap)
import tensorflow as tf
import onnxruntime as ort
import numpy as np
from ultralytics.tensorleap_folder.utils import validate_supported_models
from ultralytics.tensorleap_folder.global_params import cfg, all_clss
from code_loader.plot_functions.visualize import visualize
from code_loader.contract.datasetclasses import PredictionTypeHandler
from code_loader.inner_leap_binder.leapbinder_decorators import tensorleap_load_model, integration_test

keras_model=False # change to True if using h5
prediction_type1 = PredictionTypeHandler(name='object detection', labels=["x", "y", "w", "h"] + [cl for cl in all_clss.values()], channel_dim=1)
prediction_type2 = PredictionTypeHandler(name='concatenate_20', labels=[str(i) for i in range(20)], channel_dim=-1)
prediction_type3 = PredictionTypeHandler(name='concatenate_40', labels=[str(i) for i in range(40)], channel_dim=-1)
prediction_type4 = PredictionTypeHandler(name='concatenate_80', labels=[str(i) for i in range(80)], channel_dim=-1)

@tensorleap_load_model([prediction_type1,prediction_type2,prediction_type3,prediction_type4])
def load_model():
    model_path="/Users/yamtawachi/tensorleap/datasets/models/yolo11s.onnx"
    m_path = model_path if model_path != None else 'None_path'
    print("started custom tests")
    validate_supported_models(os.path.basename(cfg.model), m_path)
    if not os.path.exists(m_path):
        from export_model_to_tf import onnx_exporter  # TODO - currently supports only onnx
        m_path = onnx_exporter()
    keras_model = m_path.endswith(".h5")
    model = tf.keras.models.load_model(m_path) if keras_model else ort.InferenceSession(m_path)
    return model


@integration_test()
def check_custom_test_mapping(idx, subset):
    s_prepro = SamplePreprocessResponse(np.array(idx), subset)
    image = input_encoder(idx, subset)
    model = load_model()
    y_pred = model([image]) if keras_model  else model.run(None, {'images': image})
    #if subset.state != DataStateType.unlabeled: we can solve this by setting the inputs of integration_test_function(None, None) to training mode during the mapping mode.
# get gt #TODO- need to make sure that this is not needed and that if no lables it will not crash
    gt = gt_encoder(idx, subset)
    gt_img = gt_bb_decoder(image, gt)
# custom metrics
    total_loss=loss(y_pred[1],y_pred[2],y_pred[3],gt ,y_pred[0])
    cost_dic=cost(y_pred[1],y_pred[2],y_pred[3],gt)
    iou=ious(y_pred[0], s_prepro)
    conf_mat = confusion_matrix_metric(y_pred[0], s_prepro)
    # metadata
    meta_data=metadata_per_img(idx, subset)
    # vis
    img_vis=image_visualizer(image)
    pred_img=bb_decoder(image,y_pred[0])
    visualize(img_vis)
    visualize(pred_img)
    # if subset.state != DataStateType.unlabeled: #TODO- need to make sure that this is not needed and that if no lables it will not crash
    visualize(gt_img)



if __name__ == '__main__':
    # keras_model, m_path = load_and_download_model()
    check_custom_test_mapping(0, preprocess_unlabeled_func_leap())
    check_custom_test_mapping(0, preprocess_func_leap()[1])