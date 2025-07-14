import os
from code_loader.contract.datasetclasses import SamplePreprocessResponse
from code_loader.contract.enums import DataStateType
from leap_binder import (input_encoder, preprocess_func_leap, gt_encoder,
                         leap_binder, loss, gt_bb_decoder, image_visualizer, bb_decoder,
                         cost, metadata_per_img, ious, confusion_matrix_metric)
import tensorflow as tf
import onnxruntime as ort
import numpy as np
from code_loader.helpers import visualize
from ultralytics.tensorleap_folder.utils import extract_mapping, validate_supported_models
from ultralytics.tensorleap_folder.global_params import cfg


def check_custom_test():
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


if __name__ == '__main__':
    check_generic = True
    plot_vis= False
    model_path = None  # Choose None if only pt version available else, use your h5/onnx model's path.
    mapping_version = None # Set as  None if the model's name is supported by ultralytics. Else, set to the base yolo architecture name (e.x if your trained model has the same architecture as yolov11s set mapping_version=yolov11s ) .
    check_custom_test()
