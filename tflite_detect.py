# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""label_image for tflite."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np

import cv2
from tensorflow.lite.python.interpreter import Interpreter

import random

import os
import time

def load_labels(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f.readlines()]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i',
        '--input_path',
        default='/home/tnc/PycharmProjects/DATA/object_detection/poker_rect/images/test_2018-01-20',
        #default='/home/wegn/PycharmProjects/yolov5/data/images',
        help='image to be classified')
    parser.add_argument(
        '-m',
        '--model_file',
        default='/home/tnc/PycharmProjects/yolov5/runs/train/poker_ls_s640/weights/best-fp16.tflite',
        #default='/home/wegn/PycharmProjects/yolov5/yolov5s-fp16.tflite',
        help='.tflite model to be executed')
    parser.add_argument(
        '-l',
        '--label_file',
        default='/tmp/labels.txt',
        help='name of file containing labels')

    args = parser.parse_args()

    interpreter = Interpreter(model_path=args.model_file)
    interpreter.allocate_tensors()
    # [{'name': 'normalized_input_image_tensor', 'index': 357, 'shape': array([1, 300, 300, 3], dtype=int32), 'dtype': <class 'numpy.float32'>, 'quantization': (0.0, 0)}]

    # [{'name': 'raw_outputs/box_encodings', 'index': 358, 'shape': array([1, 1917, 4], dtype=int32), 'dtype': <class 'numpy.float32'>, 'quantization': (0.0, 0)},
    # {'name': 'raw_outputs/class_predictions', 'index': 359, 'shape': array([1, 1917, 13], dtype=int32), 'dtype': <class 'numpy.float32'>, 'quantization': (0.0, 0)}]

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("input_details:",input_details)
    print("output_details:",output_details)
    """
    input_details: [{'name': 'input_1', 'index': 0, 'shape': array([  1, 640, 640,   3], dtype=int32), 'shape_signature': array([  1, 640, 640,   3], dtype=int32), 'dtype': <class 'numpy.float32'>, 'quantization': (0.0, 0), 'quantization_parameters': {'scales': array([], dtype=float32), 'zero_points': array([], dtype=int32), 'quantized_dimension': 0}, 'sparsity_parameters': {}}]
    output_details: [{'name': 'Identity', 'index': 405, 'shape': array([    1, 25200,     6], dtype=int32), 'shape_signature': array([    1, 25200,     6], dtype=int32), 'dtype': <class 'numpy.float32'>, 'quantization': (0.0, 0), 'quantization_parameters': {'scales': array([], dtype=float32), 'zero_points': array([], dtype=int32), 'quantized_dimension': 0}, 'sparsity_parameters': {}}]
    """
    # check the type of the input tensor
    is_floating_model = input_details[0]['dtype'] == np.float32

    # NxHxWxC, H:1, W:2
    height ,width,channel= input_details[0]['shape'][1],input_details[0]['shape'][2],input_details[0]['shape'][3]
    #width = input_details[0]['shape'][2]
    print("input_detail:", height, width,channel, is_floating_model)

    #img = Image.open(args.image).resize((width, height))
    # add N dim
    #input_data = np.expand_dims(img, axis=0)

    images_path = os.listdir(args.input_path)
    if len(images_path)!=0:
        str_time=time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        out_path=args.input_path+'_out_'+str_time
        os.makedirs(out_path)
    else:

        images_path.append('/home/wegn/PycharmProjects/datasets/crane/images/val/2021_10_27_15_44_0.jpg')

    for pure_per_image in images_path:
        per_image=os.path.join(args.input_path,pure_per_image)
        print(pure_per_image)


        org_image = cv2.imread(per_image, 1)
        #cv2.imshow("org",org_image)
        # image=cv2.resize(image1,(int(1280*0.5),int(720*0.5)))
        #image = image[300:900, 300:900]  # height,width
        #col_resize_image = cv2.resize(org_image, (1080, 1080))

        #resize_roi = org_image[0:720, 460:1280]  # height,width

        #tongzi512
        #col_resize_image = cv2.resize(org_image,(1066,800))
        #resize_roi_bg_x,resize_roi_bg_y=133,0
        #resize_roi=col_resize_image[resize_roi_bg_y:800,resize_roi_bg_x:933]

        #paigow
        col_resize_image = cv2.resize(org_image, (1280, 720))
        resize_roi_bg_x, resize_roi_bg_y = 200, 0
        resize_roi = col_resize_image[resize_roi_bg_y:720, resize_roi_bg_x:920]
        image = cv2.resize(resize_roi, (640, 640))

        #poker_rect*********************************************************************************
        #col_resize_image = cv2.resize(org_image, (1280, 731))
        #resize_roi_bg_x, resize_roi_bg_y = 0, 0
        #resize_roi = col_resize_image[resize_roi_bg_y:int(col_resize_image.shape[0]*1.0), resize_roi_bg_x:int(col_resize_image.shape[1]*1.0)]
        #image = cv2.resize(resize_roi, (672, 384))
        ###############################################################################################

        #cv2.imshow("img",image)
        #cv2.waitKey(10)


        width_resize_scale= float(image.shape[0]) / resize_roi.shape[0]
        height_resize_scale= float(image.shape[1]) / resize_roi.shape[1]




        image_np = np.array(image).reshape((image.shape[0], image.shape[1], 3)).astype(np.uint8)  # channel:1
        # cv2.imshow('in', image)
        # cv2.waitKey(100)
        img_height, img_width = image.shape[0], image.shape[1]

        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        if is_floating_model:
            input_data = (np.float32(image_np_expanded) ) / 255.0
            #input_data = (np.float32(image_np_expanded) - args.input_mean) / args.input_std
        else:#int8
            print("int8")
            input_data = (np.float32(image_np_expanded)) / 255.0
            scale, zero_point = input_details[0]['quantization']
            input_data = (input_data / scale + zero_point).astype(np.uint8)  # de-scale

        interpreter.set_tensor(input_details[0]['index'], input_data)



        interpreter.invoke()



        preds = np.squeeze(interpreter.get_tensor(output_details[0]['index']))
        #print(input_details[0]['dtype'])
        if input_details[0]['dtype']==np.uint8:
            print("113")
            scale, zero_point = output_details[0]['quantization']
            preds = (preds.astype(np.float32) - zero_point) * scale  # re-scale
        boxes=[]
        scores=[]
        class_idxes=[]

        for pred in preds:
            max_idx=list(pred).index(max(pred[5:],key=abs))

            if pred[4]>0.4 and pred[max_idx]>0.4:
                x=(pred[0]-pred[2]/2.0)*img_width
                y=(pred[1]-pred[3]/2.0)*img_height
                width=pred[2]*img_width
                height=pred[3]*img_height
                boxes.append([x,y,width,height])
                scores.append(float(pred[max_idx]))
                class_idxes.append(max_idx-5)#前5为x,y,width,height,confidence

        nms_box_idx = cv2.dnn.NMSBoxes(boxes,scores,0.4,0.5)

        for idx,nms_id in enumerate(nms_box_idx):
            per_class=class_idxes[nms_id]+1
            box=boxes[nms_id]
            score=scores[nms_id]
            msg=str(per_class)+":"+str(score)

            box[0] /= width_resize_scale
            box[1] /= height_resize_scale
            #box[0] += resize_roi_bg_x
            #box[1] += resize_roi_bg_y


            box[2] /= width_resize_scale
            box[3] /= height_resize_scale

            color=(random.randint(0,255),random.randint(0,255),random.randint(0,255))

            cv2.putText(resize_roi,msg[:7],(int(box[0]),int(box[1]-10)),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            cv2.rectangle(resize_roi,(int(box[0]),int(box[1])),(int(box[0]+box[2]),int(box[1]+box[3])) ,color, 1, 4)
        #cv2.imshow("yolov5", col_resize_image)
        #cv2.waitKey(10)
        per_image_out_path = os.path.join(out_path, pure_per_image)
        cv2.imwrite(per_image_out_path, resize_roi)




