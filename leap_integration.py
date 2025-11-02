from ultralytics.tensorleap_folder.get_cfg import cfg
from registry import leap_binder,global_params,utils
import os
from code_loader.contract.datasetclasses import SamplePreprocessResponse
import onnxruntime as ort
import numpy as np
from code_loader.plot_functions.visualize import visualize
from code_loader.contract.datasetclasses import PredictionTypeHandler
from code_loader.inner_leap_binder.leapbinder_decorators import tensorleap_load_model, tensorleap_integration_test


if cfg.task=="pose":
    labels = []
    for k in range(17):
        labels.append(f"x_{k}")
        labels.append(f"y_{k}")
        labels.append(f"v_{k}")
    prediction_type1 = PredictionTypeHandler('output', labels=["x", "y", "w", "h", "0"] + labels, channel_dim=1)
    prediction_type2 = PredictionTypeHandler('feat_a', labels=[str(i) for i in range(65)], channel_dim=1)
    prediction_type3 = PredictionTypeHandler('feat_b', labels=[str(i) for i in range(65)], channel_dim=1)
    prediction_type4 = PredictionTypeHandler('feat_c', labels=[str(i) for i in range(65)], channel_dim=1)
    prediction_type5 = PredictionTypeHandler(name="key_points", labels=labels, channel_dim=1)
    all_predictions=[prediction_type1, prediction_type2, prediction_type3, prediction_type4, prediction_type5]
else:
    prediction_type1 = PredictionTypeHandler(name='object detection',
                                             labels=["x", "y", "w", "h"] + [cl for cl in global_params.all_clss.values()],
                                             channel_dim=1)
    prediction_type2 = PredictionTypeHandler(name='concatenate_20', labels=[str(i) for i in range(20)],
                                             channel_dim=-1)
    prediction_type3 = PredictionTypeHandler(name='concatenate_40', labels=[str(i) for i in range(40)],
                                             channel_dim=-1)
    prediction_type4 = PredictionTypeHandler(name='concatenate_80', labels=[str(i) for i in range(80)],
                                             channel_dim=-1)
    all_predictions=[prediction_type1, prediction_type2, prediction_type3, prediction_type4]


@tensorleap_load_model(all_predictions)
def load_model():
    m_path = model_path if model_path != None else 'None_path'
    print("started custom tests")
    utils.validate_supported_models(os.path.basename(cfg.model), m_path)
    if not os.path.exists(m_path):
        from export_model_to_tf import onnx_exporter  # TODO - currently supports only onnx
        m_path = onnx_exporter()
    model = ort.InferenceSession(m_path)
    return model


@tensorleap_integration_test()
def check_custom_test_mapping(idx, subset):
    s_prepro = SamplePreprocessResponse(np.array(idx), subset)
    image = leap_binder.input_encoder(idx, subset)
    model = load_model()
    y_pred = model.run(None, {'images': image})
    # get gt
    gt = leap_binder.gt_encoder(idx, subset)#
    # custom metrics
    total_loss=leap_binder.loss(*y_pred,gt)#
    cost_dic=leap_binder.cost(*y_pred,gt)#
    matrices = leap_binder.get_matrices(*y_pred, s_prepro)  #

    if not cfg.task=="pose":#
        # metadata
        meta_data=leap_binder.metadata_per_img(idx, subset)#
    # vis
    img_vis=leap_binder.image_visualizer(image)#
    pred_img=leap_binder.pred_visualizer(image,*y_pred,s_prepro) #
    gt_img = leap_binder.gt_visualizer(image, gt, data=s_prepro)#
    visualize(img_vis)#
    visualize(pred_img)#
    visualize(gt_img)#



if __name__ == '__main__':
    model_path="/Users/yamtawachi/tensorleap/ultralytics/yolo11s.onnx"
    check_custom_test_mapping(0, leap_binder.preprocess_func_leap()[1])
    check_custom_test_mapping(0, leap_binder.preprocess_func_leap()[2])