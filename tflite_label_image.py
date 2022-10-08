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
#from object_detection.anchors_data import anchors

from PIL import Image
import cv2
from tensorflow.lite.python.interpreter import Interpreter

import math
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
        #default='/home/tnc/PycharmProjects/DATA/object_detection/tongzi/test/images/27.jpg',
        default='/media/tnc/6A68482A6847F37D/object_detection/tongzi/dot061103',
        #default='/home/tnc/PycharmProjects/DATA/object_detection/shuffle_poker/images/2020-10-26-10-43-57_0002.jpg',
        #'/home/tnc/tensorflow-master/models-2.2/research/object_detection/test_images/shuffle/2020-07-04-10-35-54_0118.jpg',
        help='image to be classified')
    parser.add_argument(
        '-m',
        '--model_file',
        default='/home/tnc/PycharmProjects/yolov5/runs/train/crane6/weights/best-fp16.tflite',
        #'/home/tnc/PycharmProjects/object_detection/tongzi_20210610/frozen_graph.tflite',
        help='.tflite model to be executed')
    parser.add_argument(
        '-l',
        '--label_file',
        default='/tmp/labels.txt',
        help='name of file containing labels')
    parser.add_argument('--input_mean', default=127.5,
                        help='input_mean')
    parser.add_argument('--input_std', default=127.5,
                        help='input standard deviation')
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

    interpreter = tf.lite.Interpreter(model_path=w)  # load TFLite model
            interpreter.allocate_tensors()  # allocate
            input_details = interpreter.get_input_details()  # inputs
            output_details = interpreter.get_output_details()  # outputs
            int8 = input_details[0]['dtype'] == np.uint8  # is TFLite quantized uint8 model
    """


    # check the type of the input tensor
    floating_model = input_details[0]['dtype'] == np.float32

    # NxHxWxC, H:1, W:2
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]
    print("input_detail:",height,width,floating_model)
    #img = Image.open(args.image).resize((width, height))
    # add N dim
    #input_data = np.expand_dims(img, axis=0)
    #images_path=[]#os.listdir(args.input_path)
    images_path = os.listdir(args.input_path)
    if len(images_path)!=0:
        str_time=time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        out_path=args.input_path+'_out_'+str_time
        os.makedirs(out_path)
    else:
        images_path.append('/home/tnc/PycharmProjects/datasets/crane/images/val/2021_10_27_15_44_20.jpg')

    for pure_per_image in images_path:
        per_image=os.path.join(args.input_path,pure_per_image)
        print(pure_per_image)


        org_image = cv2.imread(per_image, 1)

        # image=cv2.resize(image1,(int(1280*0.5),int(720*0.5)))
        #image = image[300:900, 300:900]  # height,width
        col_resize_image = cv2.resize(org_image, (1080, 1080))

        col_resize_roi = col_resize_image[100:900, 100:900]  # height,width

        image = cv2.resize(col_resize_roi, (640, 640))

        width_resize_scale=float(image.shape[0])/col_resize_roi.shape[0]
        height_resize_scale=float(image.shape[1])/col_resize_roi.shape[1]

        image_np = np.array(image).reshape((image.shape[0], image.shape[1], 3)).astype(np.uint8)  # channel:1
        # cv2.imshow('in', image)
        # cv2.waitKey(100)
        im_height, im_width = image.shape[0], image.shape[1]
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        if floating_model:
            input_data = (np.float32(image_np_expanded) - args.input_mean) / args.input_std

        interpreter.set_tensor(input_details[0]['index'], input_data)

        interpreter.invoke()

        boxes = np.squeeze(interpreter.get_tensor(output_details[0]['index']))
        #scores = np.squeeze(interpreter.get_tensor(output_details[1]['index']))

        print("box:",boxes.shape)

        y_scale = 10.0
        x_scale = 10.018205052155
        height_scale = 5.0
        width_scale = 5.0

        is_poker=False
        poker_color_dict = {0: "hei", 1: 'hong', 2: 'mei', 3: 'fang'}
        poker_character_dict={1:'A',2:'2',3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9',10:'10',11:'J',12:'Q',13:'K'}
        if scores.shape[1]>=53:
            is_poker=True


        for i in range(scores.shape[0]):
            max_scor = 0.0
            opt_idx = -1
            for idx, scor in enumerate(scores[i]):
                if (scor > max_scor):
                    max_scor = scor
                    opt_idx = idx
            if (max_scor > 0.4):
                #print(scores[i])
                '''
                ycenter_a, xcenter_a, ha, wa = anchors.get_center_coordinates_and_sizes()

                    ty, tx, th, tw = tf.unstack(tf.transpose(rel_codes))
                    if self._scale_factors:
                      ty /= self._scale_factors[0]
                      tx /= self._scale_factors[1]
                      th /= self._scale_factors[2]
                      tw /= self._scale_factors[3]
                    w = tf.exp(tw) * wa
                    h = tf.exp(th) * ha
                    ycenter = ty * ha + ycenter_a
                    xcenter = tx * wa + xcenter_a
                    ymin = ycenter - h / 2.
                    xmin = xcenter - w / 2.
                    ymax = ycenter + h / 2.
                    xmax = xcenter + w / 2.
                '''

                ycenter_a, xcenter_a, ha, wa =1,1,1,1  #anchors[i]
                ty, tx, th, tw = boxes[i]
                ty /= y_scale
                tx /= x_scale
                th /= height_scale
                tw /= width_scale
                w = math.exp(tw) * wa
                h = math.exp(th) * ha
                ycenter = ty * ha + ycenter_a
                xcenter = tx * wa + xcenter_a
                ymin = ycenter - h / 2.
                xmin = xcenter - w / 2.
                ymax = ycenter + h / 2.
                xmax = xcenter + w / 2.

                (left, right, top, bottom) = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)
                (left, right, top, bottom) = (left/width_resize_scale, right/width_resize_scale,
                                              top/height_resize_scale, bottom/height_resize_scale)
                cv2.rectangle(col_resize_roi, (int(left), int(top)), (int(right), int(bottom)), (0, 0, 0), 1, 4)
                cls_str=str(opt_idx)
                if is_poker:
                    character=(opt_idx-1)%13+1#1,2,...13
                    color=int((opt_idx-1)/13)#0,1,2,3
                    #print(opt_idx,character,color)
                    cls_str=str(poker_color_dict[color])+''+str(poker_character_dict[character])
                    #print(opt_idx,cls_str)

                text = str(cls_str) + ":" + str(max_scor)[:4]

                print('score:',text)
                cv2.putText(col_resize_roi, text, (int(left), int(top - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                print("box:", round(left,1), round(top,1), round(right-left,2), round(bottom-top,2))

        #cv2.imshow(str(pure_per_image), image)
        #cv2.waitKey(0)
        per_image_out_path=os.path.join(out_path,pure_per_image)
        cv2.imwrite(per_image_out_path,col_resize_roi)




