# -*- coding: UTF-8 -*-

import tensorflow as tf

import numpy as np

input_arrays = ["normalized_input_image_tensor"]#image_tensor
output_arrays = ["raw_outputs/box_encodings","raw_outputs/class_predictions"]

'''
def convert_tflite_pb_to_tflite(input_tflite_pb_path,out_path):
    #graph_def_file = " /quant/tflite_graph.pb"
    input_arrays = ["normalized_input_image_tensor"]
    output_arrays = ['TFLite_Detection_PostProcess', 'TFLite_Detection_PostProcess:1', 'TFLite_Detection_PostProcess:2',
                     'TFLite_Detection_PostProcess:3']
    input_tensor = {"normalized_input_image_tensor": [1, 300, 300, 3]}

    converter = tf.lite.TFLiteConverter.from_frozen_graph(input_tflite_pb_path, input_arrays, output_arrays, input_tensor)
    converter.allow_custom_ops = True
    tflite_model = converter.convert()
    open(out_path, "wb").write(tflite_model)
'''

# pb file to tflite file
def convert_tflite_model(graph_input_complete_path,graph_output_path):

    print_node(graph_input_complete_path)

    converter = tf.lite.TFLiteConverter.from_frozen_graph(
        graph_input_complete_path, input_arrays, output_arrays)
    tflite_model = converter.convert()
    open(graph_output_path, "wb").write(tflite_model)
    print('tflite graph have writen in %s' % (graph_output_path))




def quant_f16_tflite_model(graph_input_cnmplete_path,graph_output_path):
    converter = tf.lite.TFLiteConverter.from_frozen_graph(
        graph_input_cnmplete_path,input_arrays, output_arrays)

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    #converter.target_spec.supported_types = [tf.lite.constants.FLOAT16]
    tflite_quant_model = converter.convert()
    open(graph_output_path, "wb").write(tflite_quant_model)
    print('tflite graph have writen in %s' % (graph_output_path))

def quant_yolov5_2_f16_tflite_model(model,tflite_model_fp16_file):
    converter = tf.lite.TFLiteConverter.from_saved_model(model)

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]

    tflite_model = converter.convert()
    #tflite_model_fp16_file.write_bytes(tflite_model)
    open(tflite_model_fp16_file, "wb").write(tflite_model)
    print('tflite graph have writen in %s' % (tflite_model_fp16_file))

def yolov5_judge_support_ops(model_path,save_dir):
    converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                           tf.lite.OpsSet.SELECT_TF_OPS]
    tflite_model = converter.convert()
    open(save_dir, "wb").write(tflite_model)


def print_node(frozen_graph_filename):
    graph = tf.GraphDef()

    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph.ParseFromString(f.read())

    for ind,val in enumerate(graph.node):
        print(ind,val.name,val.op)
        [print(u'└─── %d ─ %s' % (i, n)) for i, n in enumerate(val.input)]

def represnetvtive_dataset():
    for _ in range(100):
        data=np.random.rand(1,512,512,3)
        yield  [data.astype(np.float32)]
def convert_to_uint8_model(model_path,save_path):
    converter=tf.lite.TFLiteConverter.from_saved_model(model_path)
    converter.optimizations=[tf.lite.Optimize.DEFAULT]
    converter.representative_dataset=represnetvtive_dataset
    converter.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type=tf.uint8
    converter.inference_output_type=tf.uint8
    tflite_quant_model=converter.convert()
    open(save_path,"wb").write(tflite_quant_model)





if __name__ == '__main__':
    graph_def_file = '/home/tnc/PycharmProjects/classify/model_paijiu_gray_c21_s100x100_mbnetv1_20220226/frozen_graph.pb'

    out_graph_file = '/home/tnc/PycharmProjects/classify/model_paijiu_gray_c21_s100x100_mbnetv1_20220226/frozen_graph.tflite'

    # graph_def_file = "/home/tnc/tensorflow-master/models-1.12/self/mobilenet_v1_1.0_224/mobilenet_v1_1.0_224_frozen.pb"
    # out_graph_file="/home/tnc/tensorflow-master/models-1.12/self/mobilenet_v1_1.0_224/mbnetv1_s224_082902.tflite"
    convert_tflite_model(graph_def_file,out_graph_file)

    quant_f16_in_graph_file='/Users/wegn/PycharmProjects/yolov5/runs/train/crane/weights/best_saved_model'
    quant_f16_out_graph_file = '/Users/wegn/PycharmProjects/yolov5/runs/train/crane/weights/best-fp32--z.tflite'
    #quant_yolov5_2_f16_tflite_model(quant_f16_in_graph_file,quant_f16_out_graph_file)
    #yolov5_judge_support_ops(quant_f16_in_graph_file,quant_f16_out_graph_file)


    tmp='/home/tnc/PycharmProjects/yolov5/runs/train/shuffle_poker_Ln_s416/weights/best_saved_model'
    tmp_out='/home/tnc/PycharmProjects/yolov5/runs/train/shuffle_poker_Ln_s416/weights/best_fp16.tflite'
    #convert_to_uint8_model(tmp,tmp_out)
    quant_yolov5_2_f16_tflite_model(tmp,tmp_out)





