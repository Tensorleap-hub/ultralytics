import os

from code_loader.contract.datasetclasses import SamplePreprocessResponse, PredictionTypeHandler
from code_loader.plot_functions.visualize import visualize
from ultralytics.tensorleap_folder.pose.leap_binder import (input_encoder, preprocess_func_leap, gt_encoder,
                        loss, image_visualizer,
                         cost, draw_skeleton, get_matrices,
                         draw_gt_skeleton)
import onnxruntime as ort
from ultralytics.tensorleap_folder.pose.utils import  set_leap_yaml2root, validate_supported_models
from ultralytics.tensorleap_folder.pose.global_params import cfg
from code_loader.inner_leap_binder.leapbinder_decorators import tensorleap_load_model, tensorleap_integration_test
labels = []
for k in range(17):
    labels.append(f"x_{k}")
    labels.append(f"y_{k}")
    labels.append(f"v_{k}")

prediction_type1 = PredictionTypeHandler('output', labels = ["x", "y", "w", "h", "0"] + labels, channel_dim=1)
prediction_type2 = PredictionTypeHandler('feat_a', labels=[str(i) for i in range(65)], channel_dim=1)
prediction_type3 = PredictionTypeHandler('feat_b', labels=[str(i) for i in range(65)], channel_dim=1)
prediction_type4 = PredictionTypeHandler('feat_c', labels=[str(i) for i in range(65)], channel_dim=1)
prediction_type5 = PredictionTypeHandler(name="key_points", labels=labels, channel_dim=1)


@tensorleap_load_model([prediction_type1, prediction_type2, prediction_type3, prediction_type4, prediction_type5])
def load_model():
    m_path= model_path if model_path!=None else 'None_path'
    validate_supported_models(os.path.basename(cfg.model),m_path)
    if not os.path.exists(m_path):
        from ultralytics.tensorleap_folder.pose.export_model_to_tf import onnx_exporter #TODO - currently supports only onnx
        m_path=onnx_exporter()
    model = ort.InferenceSession(m_path)
    return model


@tensorleap_integration_test()
def check_custom_integration(idx, subset):
    model = load_model()
    # get input images
    image = input_encoder(idx, subset)
    # predict
    y_pred = model.run(None, {'images': image})

    # get gt
    gt = gt_encoder(idx, subset)
    # custom metrics
    total_loss_0=loss(y_pred[1],y_pred[2],y_pred[3], y_pred[4],
                    gt, y_pred[0])
    s_prepro = SamplePreprocessResponse(idx, subset)

    # vis
    annotated_bgr = draw_skeleton(image,y_pred[0], y_pred[1], y_pred[2], y_pred[3], y_pred[4], data=s_prepro)  # BGR ndarray
    # gt_bgr = draw_gt_on_image(image, gt, data=s_prepro)
    gt_bgr = draw_gt_skeleton(image, gt, data=s_prepro)
    img_vis = image_visualizer(image)

    visualize(annotated_bgr)
    visualize(gt_bgr)
    visualize(img_vis)

    # matrices
    mats = get_matrices(y_pred[0], y_pred[1], y_pred[2], y_pred[3], y_pred[4], preprocess=s_prepro)
    cost_dic = cost(y_pred[1], y_pred[2], y_pred[3], y_pred[4], gt)



if __name__ == '__main__':
    model_path =None# '/Users/yamtawachi/tensorleap/ultralytics/yolo11s-pose.onnx'  # Choose None if only pt version available else, use your h5/onnx model's path.
    set_leap_yaml2root(cfg)
    check_custom_integration(0, preprocess_func_leap()[0])