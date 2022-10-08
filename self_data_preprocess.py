#from lxml import etree
import tensorflow as tf
import xml.etree.ElementTree as ET
import os
import cv2
import numpy as np
import json

import shutil
import random
resize_radio=1#0.375


def rename_xml_filename(xmls_path,out_path):
    xmls_file = os.listdir(xmls_path)
    for per_xml in xmls_file:
        print(per_xml[:per_xml.find('.')])
        readTree = ET.ElementTree(file=xmls_path + '/' + per_xml)
        readRoot = readTree.getroot()
        for node in readRoot:
            #print(node.tag)
            if node.tag=="filename":
                node.text=per_xml[:per_xml.find('.')]+'.jpg'
        readTree.write(out_path+'/'+per_xml)




def cvt_xml(sig_class_path,label,out_path):
    xml_path=os.path.join(sig_class_path+'/'+label+'.xml')
    if os.path.exists(xml_path)==False:
        print('dont exist',xml_path)
        os._exit(0)
    os.makedirs(out_path+'/image')
    os.makedirs(out_path+'/xml')

    image_out_path=os.path.join(out_path+'/image')
    xml_out_path=os.path.join(out_path+'/xml')

    readTree=ET.ElementTree(file=xml_path)
    #print(readTree)
    readRoot=readTree.getroot()
    #print(readRoot.tag)#dataset
    trainvalTxt=open(out_path+'/'+'trainval.txt','w')



    for image_file in readTree.iter(tag='image'):
        #print('image:',len(image_file),image_file.tag,image_file.attrib['file'])
        if len(image_file)>0:#存在box
            trainvalTxt.write(label+'_'+image_file.attrib['file'][:-5] + '\n')


            writeRoot=ET.Element('annotation')

            writeFolder=ET.SubElement(writeRoot,'folder')
            writeFolder.text='tongzi'

            #print('path:',sig_class_path+'/'+image_file.attrib['file'])
            img=cv2.imread(sig_class_path+'/'+image_file.attrib['file'],0)
            if not img.any():
                print('dont exist',image_file.attrib['file'])
                os._exit(0)
            #cv2.imshow('sd',img)
            #cv2.waitKey(0)
            dst=cv2.resize(img,(0,0),None,resize_radio,resize_radio)

            image_shape=dst.shape

            cv2.imwrite(image_out_path+'/'+label+'_'+image_file.attrib['file'][:-5]+'.jpeg',dst)

            writeFilename = ET.SubElement(writeRoot, 'filename')
            writeFilename.text = label+'_'+image_file.attrib['file'][:-5] + '.jpeg'

            writeSize = ET.SubElement(writeRoot, 'size')
            sizeWidth = ET.SubElement(writeSize, 'height')
            sizeWidth.text =str(image_shape[0])
            sizeHeigh = ET.SubElement(writeSize, 'width')
            sizeHeigh.text = str(image_shape[1])
            sizeDepth = ET.SubElement(writeSize, 'depth')
            sizeDepth.text = '1'#image_shape[2]

            writeSegmented = ET.SubElement(writeRoot, 'segmented')
            writeSegmented.text='0'

            for idx,box in enumerate(image_file):
                if True:#为了去重
                #if len(image_file)==1 or (idx > 0 and image_file[idx].attrib['left']!= image_file[idx-1].attrib['left']):
                    writeObject = ET.SubElement(writeRoot, 'object')
                    objectName = ET.SubElement(writeObject, 'name')
                    objectName.text = label
                    objectPose = ET.SubElement(writeObject, 'pose')
                    objectPose.text = '0'
                    objectTruncated = ET.SubElement(writeObject, 'truncated')
                    objectTruncated.text = '0'
                    objectDifficult = ET.SubElement(writeObject, 'difficult')
                    objectDifficult.text = '0'
                    objectBndbox = ET.SubElement(writeObject, 'bndbox')

                    bndboxXmin = ET.SubElement(objectBndbox, 'xmin')
                    bndboxXmin.text = str(int(float(box.attrib['left'])*resize_radio))#box.attrib['left']
                    bndboxYmin = ET.SubElement(objectBndbox, 'ymin')
                    bndboxYmin.text = str(int(float(box.attrib['top'])*resize_radio))#box.attrib['top']
                    bndboxXmax = ET.SubElement(objectBndbox, 'xmax')
                    bndboxXmax.text = str(int(float(box.attrib['left'])*resize_radio +
                                              float(box.attrib['width'])*resize_radio))
                    #str(int(box.attrib['left']) + int(box.attrib['width']))
                    bndboxYmax = ET.SubElement(objectBndbox, 'ymax')
                    bndboxYmax.text = str(int(float(box.attrib['top'])*resize_radio +
                                              float(box.attrib['height'])*resize_radio))
                    #str(int(box.attrib['top']) + int(box.attrib['height']))

            writeTree=ET.ElementTree(writeRoot)
            writeTree.write(xml_out_path+'/'+label+'_'+image_file.attrib['file'][:-5]+'.xml')
    trainvalTxt.close()


def iter_obj(trainVal_path,out_path):
    obj_path=os.listdir(trainVal_path)
    for sig_obj in obj_path:
        print(sig_obj)
        os.makedirs(out_path+sig_obj)

        label=sig_obj
        sig_obj_path=os.path.join(trainVal_path+sig_obj)
        sig_out_path=out_path+sig_obj
        cvt_xml(sig_obj_path,label,sig_out_path)


def create_trainval_txt_and_calc_aspect_radio(input_path,output_path):
    annotations_path = os.path.join(input_path +'/'+ 'annotations')
    xml_path = os.path.join(annotations_path +'/'+ 'xmls')
    images_path = os.path.join(input_path +'/'+ 'images')


    images=os.listdir(images_path)
    #images = os.listdir(xml_path)
    trainvalTxt = open(annotations_path + '/' + 'trainval.txt', 'a')
    for per_img in images:
        trainvalTxt.write(per_img[:per_img.find('.')]+'\n') #jpg
    trainvalTxt.close()



def tmp_xml(xml_path,out_path):

    xml_file=os.listdir(xml_path)
    for per_xml in xml_file:
        readTree = ET.ElementTree(file=xml_path+'/'+per_xml)

        readTree.write(out_path+'/'+per_xml[:per_xml.find('.')]+'.xml')


#直接生成标准检测训练集
def create_object_detection_sample(input_path,output_path,resize_scale):

    annotations_path=os.path.join(output_path+'annotations')
    xml_path=os.path.join(annotations_path+'/'+'xmls')
    images_path=os.path.join(output_path+'images')
    os.makedirs(images_path)
    os.makedirs(annotations_path)
    os.makedirs(xml_path)
    is_create_success=False
    sum_aspect_radio=0.0
    box_cnt=0
    obj_path = os.listdir(input_path)
    for sig_obj in obj_path:
        print(sig_obj)
        label = sig_obj
        sig_class_path = os.path.join(input_path + sig_obj)
        sig_xml_path = os.path.join(sig_class_path + '/' + label + '.xml')

        if os.path.exists(sig_xml_path) == False:
            print('dont exist', sig_xml_path)
            os._exit(0)

        readTree = ET.ElementTree(file=sig_xml_path)
        # print(readTree)
        readRoot = readTree.getroot()
        # print(readRoot.tag)#dataset
        trainvalTxt = open(annotations_path + '/' + 'trainval.txt', 'a')

        for image_file in readTree.iter(tag='image'):
            # print('image:',len(image_file),image_file.tag,image_file.attrib['file'])
            if len(image_file) > 0:  # 存在box
                trainvalTxt.write(label + '_' + image_file.attrib['file'][:-5] + '\n')

                writeRoot = ET.Element('annotation')

                writeFolder = ET.SubElement(writeRoot, 'folder')
                writeFolder.text = 'tongzi'

                # print('path:',sig_class_path+'/'+image_file.attrib['file'])
                img = cv2.imread(sig_class_path + '/' + image_file.attrib['file'], 0)
                if not img.any():
                    print('dont exist', image_file.attrib['file'])
                    os._exit(0)
                # cv2.imshow('sd',img)
                # cv2.waitKey(0)
                dst = cv2.resize(img, (0, 0), None, resize_scale, resize_scale)

                image_shape = dst.shape

                #cv2.imwrite(images_path + '/' + label + '_' + image_file.attrib['file'][:-5] + '.jpeg', dst)

                writeFilename = ET.SubElement(writeRoot, 'filename')
                writeFilename.text = label + '_' + image_file.attrib['file'][:-5] + '.jpeg'

                writeSize = ET.SubElement(writeRoot, 'size')
                sizeWidth = ET.SubElement(writeSize, 'height')
                sizeWidth.text = str(image_shape[0])
                sizeHeigh = ET.SubElement(writeSize, 'width')
                sizeHeigh.text = str(image_shape[1])
                sizeDepth = ET.SubElement(writeSize, 'depth')
                sizeDepth.text = '1'  # image_shape[2]

                writeSegmented = ET.SubElement(writeRoot, 'segmented')
                writeSegmented.text = '0'
                is_create_success=True
                for idx, box in enumerate(image_file):
                    if True:  # 为了去重
                        # if len(image_file)==1 or (idx > 0 and image_file[idx].attrib['left']!= image_file[idx-1].attrib['left']):
                        writeObject = ET.SubElement(writeRoot, 'object')
                        objectName = ET.SubElement(writeObject, 'name')
                        objectName.text = label
                        objectPose = ET.SubElement(writeObject, 'pose')
                        objectPose.text = '0'
                        objectTruncated = ET.SubElement(writeObject, 'truncated')
                        objectTruncated.text = '0'
                        objectDifficult = ET.SubElement(writeObject, 'difficult')
                        objectDifficult.text = '0'
                        objectBndbox = ET.SubElement(writeObject, 'bndbox')

                        bndboxXmin = ET.SubElement(objectBndbox, 'xmin')
                        bndboxXmin.text = str(int(float(box.attrib['left']) * resize_scale))  # box.attrib['left']
                        bndboxYmin = ET.SubElement(objectBndbox, 'ymin')
                        bndboxYmin.text = str(int(float(box.attrib['top']) * resize_scale))  # box.attrib['top']
                        bndboxXmax = ET.SubElement(objectBndbox, 'xmax')
                        bndboxXmax.text = str(int(float(box.attrib['left']) * resize_scale +
                                                  float(box.attrib['width']) * resize_scale))
                        # str(int(box.attrib['left']) + int(box.attrib['width']))
                        bndboxYmax = ET.SubElement(objectBndbox, 'ymax')
                        bndboxYmax.text = str(int(float(box.attrib['top']) * resize_scale +
                                                  float(box.attrib['height']) * resize_scale))
                        # str(int(box.attrib['top']) + int(box.attrib['height']))
                        sum_aspect_radio+=float(box.attrib['width'])/float(box.attrib['height'])
                        box_cnt+=1


                writeTree = ET.ElementTree(writeRoot)
                writeTree.write(xml_path + '/' + label + '_' + image_file.attrib['file'][:-5] + '.xml')
        trainvalTxt.close()
    print("all aspect radio:",sum_aspect_radio/box_cnt)
    if is_create_success==False:
        shutil.rmtree(out_path)



modify_input_path='/media/tnc/BA62CCAF62CC71A5/tongzi_image'
modify_output_path='/home/tnc/PycharmProjects/DATA/object_detection/tongzi_complete'
def modify_image():
    capture_path_list=os.listdir(modify_input_path)
    for sig_cap in capture_path_list:
        if(sig_cap[-2:] == 'sg'):#此目录下的种类都单独存在一个文件夹中
            object_path=os.path.join(modify_input_path,sig_cap)
            object_path_list=os.listdir(object_path)
            for sig_obj in object_path_list:
                print(sig_obj)

                sig_obj_data_name=sig_obj[:sig_obj.find('_')]#eg.2017-07-27-11-31
                sig_obj_tz_name=sig_obj[sig_obj.rfind('_')+1:]##1,2,3,4,5,6,7,8,9,10
                #print('tz:',sig_obj_tz_name)
                #print('name:',sig_obj_data_name)
                sig_obj_path=os.path.join(object_path,sig_obj)
                sig_obj_out_path= modify_output_path+'/tz_'+sig_obj_tz_name
                #if os.path.exists(modify_output_path+'/tz_'+sig_obj[-1]):
                #    continue
                #else:
                os.makedirs( modify_output_path+'/tz_'+sig_obj_tz_name,exist_ok=True)

                #修改image
                image_path_list=os.listdir(sig_obj_path)
                for sig_image in image_path_list:
                    sig_image_path=os.path.join(sig_obj_path,sig_image)
                    if not os.path.exists(sig_image_path):
                        print(sig_image_path,'dont exist!')
                        os._exists(0)
                    image_data = cv2.imread(sig_image_path, 0)
                    #if not img.any():
                    #    print('dont exist', image_file.attrib['file'])
                    #    os._exit(0)
                    # cv2.imshow('sd',img)
                    # cv2.waitKey(0)
                    #dst = cv2.resize(img, (0, 0), None, resize_radio, resize_radio)
                    #image_shape = dst.shape
                    cv2.imwrite(sig_obj_out_path + '/' +sig_obj_data_name+'_'+sig_image[:4]+'.jpeg', image_data)

                #修改xml
                sig_obj_xml_path=sig_obj_path+'/'+sig_obj_data_name+'.xml'
                if os.path.exists(sig_obj_xml_path) == False:
                    print(sig_obj_xml_path,'dont exist!')
                    os._exit(0)
                #readTree = ET.ElementTree(file=sig_obj_xml_path)
                # print(readTree)
                #readRoot = readTree.getroot()
                #for image_file in readTree.iter(tag='image'):
                #    print(image_file.attrib['file'])
                tree = ET.ElementTree()
                tree.parse(sig_obj_xml_path)

                nodes=tree.findall('images/image')
                #result_nodes = []
                #for node in nodes:
                #    if if_match(node, kv_map):
                #        result_nodes.append(node)
                for node in nodes:
                    idx=node.attrib['file']
                    sig_image_name=sig_obj_data_name+'_'+idx[:-4]+'.jpeg'
                    key_map={'file':sig_image_name}
                    for key in key_map:
                        node.set(key, key_map.get(key))
                #for node in nodes:
                #    print(node.attrib['file'])

                tree.write(sig_obj_out_path+'/'+sig_obj_data_name+'.xml',encoding="utf-8",xml_declaration=True)




#获取特定label的xml和image,用于训练黑红A和梅方4
def get_aim_img(input_path,output_path):
    annotations_path = os.path.join(input_path + '/annotations')
    xml_path = os.path.join(annotations_path + '/' + 'xmls')
    images_path = os.path.join(input_path + '/images')

    out_annotations_path = os.path.join(output_path + '/annotations')
    out_xml_path = os.path.join(out_annotations_path + '/' + 'xmls')
    out_images_path = os.path.join(output_path + '/images')
    os.makedirs(out_images_path)
    os.makedirs(out_annotations_path)
    os.makedirs(out_xml_path)

    xml_file = os.listdir(xml_path)
    scores = {'1': 0, '14': 0, '30': 0,'43':0}
    for per_xml in xml_file:
        #print(per_xml)
        readTree = ET.ElementTree(file=xml_path + '/' + per_xml)
        #readRoot = readTree.getroot()
        for object in readTree.iter(tag='object'):
            for eoj in object:
                #print(eoj.tag,eoj.text)
                if eoj.tag == "name":
                    if eoj.text=="1" or eoj.text=="14" or eoj.text=="30" or eoj.text=="43":
                        scores[eoj.text]+=1
                        readTree.write(out_xml_path + '/' + per_xml)
                        per_img_path=per_xml[:per_xml.find(".")]+".jpg"
                        image = cv2.imread(images_path+"/"+per_img_path, 0)
                        cv2.imwrite(out_images_path+"/"+per_img_path,image)
                        print(per_img_path)
                        break
    print(scores)

#get_xml_type:0,遍历文件夹下xml,1:读取trainvaltxt
def cheak_worng_img_n_label(image_path,get_xml_type,xml_path,trainvaltxt_path,output_path,cls_cnt):

    if get_xml_type==0:
        xml_file = os.listdir(xml_path)

    else:
        trainvaltxt = open(trainvaltxt_path, 'r')
        xml_file = trainvaltxt.readlines()
        for idx_x,str_x in enumerate(xml_file):
            xml_file[idx_x]=str_x.strip()+".xml"

    for i in range(1,cls_cnt+1):
        tmp_path=os.path.join(output_path+'/'+str(i))
        os.makedirs(tmp_path)
    #draw_imgs=[]
    #for i in range(52):
    #    img = np.zeros((720, 1280, 1), np.uint8)
    #    #img.fill(255)
    #    draw_imgs.append(img)
    sum=[]
    aspect_dict = {}
    name_dict={}
    for i in range(cls_cnt):
        sum.append(0)
    all_xml_cnt=len(xml_file)
    for xml_idx,per_xml in enumerate(xml_file):
        if(xml_idx%1000==0):
            print("calced: ",xml_idx,' of ',all_xml_cnt)
        rf_idx=per_xml.rfind("_")
        if rf_idx==-1:
            xml_time_name=per_xml[:per_xml.rfind("-")]
        else:
            xml_time_name = per_xml[:per_xml.rfind("_")]
        #print(per_xml,",",xml_time_name)
        #if (name_dict.get(xml_time_name) == None):
        #    name_dict[xml_time_name]=0
        #else:
        #    name_dict[xml_time_name] +=1
        #    continue
        readTree = ET.ElementTree(file=xml_path + '/' + per_xml)
        readRoot = readTree.getroot()
        pure_img_path=''
        for node in readRoot:
            if node.tag == "filename":
                pure_img_path=node.text
                #print(pure_img_path)
                break

        for idx,object in enumerate(readTree.iter(tag='object')):
            for eoj in object:
                #cls=''
                if eoj.tag == "name":
                    #print("name:",eoj.text)
                    cls=eoj.text
                    sum[int(cls) - 1] += 1
                elif eoj.tag=="bndbox":
                    per_img_path = image_path + "/" + pure_img_path
                    #print('ima_path:',per_img_path)
                    image = cv2.imread(per_img_path, 0)

                    im_height, im_width = image.shape[0], image.shape[1]
                    xmin, xmax, ymin, ymax = 0, 0, 0, 0
                    for box in eoj:
                        if(box.tag=="xmin"):
                            xmin=int(box.text)
                        elif(box.tag == "xmax"):
                            xmax=int(box.text)
                        elif (box.tag == "ymin"):
                            ymin=int(box.text)
                        elif (box.tag == "ymax"):
                            ymax=int(box.text)
                    #print("rect:", xmin, xmax, ymin, ymax,'(',im_height,im_width,')')
                    if (xmax < im_width and ymax < im_height):

                        roi_img = image[ymin:ymax, xmin:xmax]  # height,width

                        cv2.imwrite(output_path + '/' + cls + '/'
                                    + pure_img_path[:pure_img_path.find('.')]+"_"+str(idx)+".jpeg", roi_img)
                    aspect_radio = float(xmax - xmin)/(ymax - ymin)
                    aspect_radio = round(aspect_radio, 1)
                    if (aspect_dict.get(aspect_radio) == None):
                        aspect_dict[aspect_radio] = [1, ymax - ymin, xmax - xmin]
                    else:
                        aspect_dict[aspect_radio][0] += 1
                        aspect_dict[aspect_radio][1] += (ymax - ymin)
                        aspect_dict[aspect_radio][2] += (xmax - xmin)

    print('sum:',sum)
    for key in aspect_dict.keys():
        aspect_dict[key][1]=round(aspect_dict[key][1]/aspect_dict[key][0],1)
        aspect_dict[key][2] = round(aspect_dict[key][2] / aspect_dict[key][0], 1)
    for i in sorted(aspect_dict):
        print(i,aspect_dict[i])


def calc_aspect_radio_n_size(input_path,output_path):
    annotations_path = os.path.join(input_path +'/'+ 'annotations')
    xml_path = os.path.join(annotations_path +'/'+ 'xmls')
    #images_path = os.path.join(input_path +'/'+ 'images')
    xml_file = os.listdir(xml_path)

    #sum = []
    #for i in range(52):
        #sum.append(0)
    sum_dict={}
    aspect_dict={}#key:aspect_radio,val:[count,height_sum,width_sum]
    #aspect_dict[0]=[0,0,0]



    for idx,per_xml in enumerate(xml_file):
        if(idx%100==0):
            print(idx,"of",len(xml_file))
        readTree = ET.ElementTree(file=xml_path + '/' + per_xml)
        readRoot = readTree.getroot()
        size = readRoot.find("size")
        width = int(size.find("width").text)
        height = int(size.find("height").text)
        #if width!=height:
        #    print("mis:",idx,",",width,":",height)
        #continue

        #pure_img_path=''
        for object in readTree.iter(tag='object'):
            for eoj in object:
                #cls=''
                if eoj.tag == "name":
                    #print("name:",eoj.text)
                    cls=eoj.text
                    #sum[int(cls) - 1] += 1
                    if(sum_dict.get(cls)==None):
                        sum_dict[cls]=0
                    else:
                        sum_dict[cls]+=1
                elif eoj.tag=="bndbox":
                    #per_img_path = image_path + "/" + pure_img_path
                    #print('ima_path:',per_img_path)
                    #image = cv2.imread(per_img_path, 0)

                    #im_height, im_width = image.shape[0], image.shape[1]
                    xmin, xmax, ymin, ymax = 0, 0, 0, 0
                    for box in eoj:
                        if(box.tag=="xmin"):
                            xmin=int(box.text)
                        elif(box.tag == "xmax"):
                            xmax=int(box.text)
                        elif (box.tag == "ymin"):
                            ymin=int(box.text)
                        elif (box.tag == "ymax"):
                            ymax=int(box.text)
                    #print("rect:", xmin, xmax, ymin, ymax,cls)
                    aspect_radio=float(xmax-xmin)/(ymax-ymin)#float(xmax-xmin)/(ymax-ymin)
                    aspect_radio=round(aspect_radio,1)
                    if(aspect_dict.get(aspect_radio)==None):
                        aspect_dict[aspect_radio]=[1,ymax-ymin,xmax-xmin]
                    else:
                        aspect_dict[aspect_radio][0]+=1
                        aspect_dict[aspect_radio][1] += (ymax-ymin)
                        aspect_dict[aspect_radio][2] += (xmax - xmin)

    #print(sum)

    print(sum_dict)

    #for key,value in aspect_dict.items():
    for key in aspect_dict.keys():
        aspect_dict[key][1]=round(aspect_dict[key][1]/aspect_dict[key][0],1)
        aspect_dict[key][2] = round(aspect_dict[key][2] / aspect_dict[key][0], 1)
    for i in sorted(aspect_dict):
        print(i,aspect_dict[i])


def change_xml_cls(xml_path,out_path):
    os.makedirs(out_path)
    xml_file = os.listdir(xml_path)
    for per_xml in xml_file:
        readTree = ET.ElementTree(file=xml_path + '/' + per_xml)
        for object in readTree.iter(tag='object'):
            for eoj in object:
                #cls=''
                if eoj.tag == "name":
                    print("name:",eoj.text)
                    eoj.text=str(24)
        readTree.write(out_path + '/' + per_xml[:per_xml.find('.')] + '.xml')



def expand_img_n_change_xml(input_path,output_path):
    annotations_path = os.path.join(input_path +'/'+ 'annotations')
    xml_path = os.path.join(annotations_path +'/'+ 'xmls')
    images_path = os.path.join(input_path +'/'+ 'images')
    xml_file = os.listdir(xml_path)


    if (os.path.exists(output_path)):
        os.removedirs(output_path)
    os.makedirs(output_path)

    out_annotations_path = os.path.join(output_path + '/' + 'annotations')
    os.mkdir(out_annotations_path)
    out_xml_path = os.path.join(out_annotations_path + '/' + 'xmls')
    os.mkdir(out_xml_path)
    out_images_path = os.path.join(output_path + '/' + 'images')
    os.mkdir(out_images_path)

    aim_width,aim_height=720,720
    #print(aim_height,aim_width)
    #exp_xml_list=os.listdir('/home/tnc/PycharmProjects/DATA/object_detection/shuffle_poker/annotations/xmls')
    all_xml_cnt = len(xml_file)
    for xml_idx, per_xml in enumerate(xml_file):
        #if per_xml in exp_xml_list:
        #    continue
        if (xml_idx % 1000 == 0):
            print("calced: ", xml_idx, ' of ', all_xml_cnt)
        readTree = ET.ElementTree(file=xml_path + '/' + per_xml)
        root = readTree.getroot()  #

        size = root.find("size")
        all_width = int(size.find("width").text)
        all_height = int(size.find("height").text)
        filename = root.find("filename").text
        #print(filename)
        per_img_rect=[]
        mean_size=[0,0]
        for obj in root.findall('object'):
            bbox = obj.find('bndbox')
            # 获取bbox坐标信息
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)

            per_img_rect.append([-1,xmin,ymin,xmax,ymax])
        per_img_rect.sort(key=lambda x:x[1])
        all_xmin=per_img_rect[0][1]
        per_img_rect.sort(key=lambda x: x[2])
        all_ymin = per_img_rect[0][2]
        image=cv2.imread(images_path+"/"+filename,0)
        x_add=aim_width-all_width
        left_x_add=int(x_add/2)
        if(left_x_add>all_xmin):
            left_x_add=all_xmin
        right_x_add=x_add-left_x_add#TODO:右侧超出的情况

        y_add=aim_height-all_height
        up_y_add=int(y_add/2)
        if(up_y_add>all_ymin):
            up_y_add=all_ymin
        down_y_add=y_add-up_y_add
        #cv2.imshow("org",image)
        left_org_img=image[:,0:left_x_add]
        left_mirr_img=cv2.flip(left_org_img,1)

        #cv2.imshow("left", left_mirr_img)
        #cv2.waitKey(0)
        right_org_img=image[:,all_width-right_x_add:all_width]
        right_mirr_img=cv2.flip(right_org_img,1)
        #print(left_mirr_img.shape[0],left_mirr_img.shape[1],image.shape[0],image.shape[1])
        left_comb_img= np.hstack([left_mirr_img,image])
        right_comb_img=np.hstack([left_comb_img,right_mirr_img])
        #cv2.imshow("ox", right_comb_img)
        up_org_img=right_comb_img[0:up_y_add,:]
        up_mirr_img=cv2.flip(up_org_img,0)

        down_org_img=right_comb_img[all_height-down_y_add:all_height,:]
        down_mirr_img=cv2.flip(down_org_img,0)

        up_comb_img=np.vstack([up_mirr_img,right_comb_img])
        comb_img=np.vstack([up_comb_img,down_mirr_img])
        comb_height,comb_width = comb_img.shape[0], comb_img.shape[1]
        #cv2.imshow("df",down_comb_img)
        #cv2.waitKey(0)
        cv2.imwrite(out_images_path+'/'+filename,comb_img)

        for node in root:
            if node.tag == "filename":
                pure_img_path = node.text
                # print(pure_img_path)
                # break
            elif node.tag == "size":
                for size in node:
                    if (size.tag == "height"):
                        size.text=str(comb_height)
                    elif (size.tag == "width"):
                        size.text=str(comb_width)
        for idx, object in enumerate(readTree.iter(tag='object')):
            for eoj in object:
                if eoj.tag == "name":
                    # print("name:",eoj.text)
                    cls = eoj.text
                elif eoj.tag == "bndbox":

                    # im_height, im_width = image.shape[0], image.shape[1]
                    #xmin, xmax, ymin, ymax = 0, 0, 0, 0
                    for box in eoj:
                        if (box.tag == "xmin"):
                            xmin = int(box.text)
                            box.text=str(xmin+left_x_add)
                        elif (box.tag == "xmax"):
                            xmax = int(box.text)
                            box.text=str(xmax+left_x_add)
                        elif (box.tag == "ymin"):
                            ymin = int(box.text)
                            box.text=str(ymin+up_y_add)
                        elif (box.tag == "ymax"):
                            ymax = int(box.text)
                            box.text=str(ymax+up_y_add)
                            # print("rect:", xmin, xmax, ymin, ymax,'(',im_height,im_width,')')
        readTree.write(out_xml_path + '/' + per_xml)


def convert_annotation_to_yolov5_data(input_path,output_path,train_val_proportion):
    annotations_path = os.path.join(input_path +'/'+ 'annotations')
    xml_path = os.path.join(annotations_path +'/'+ 'xmls')
    all_images_path = os.path.join(input_path +'/'+ 'images/train')

    xml_file_list = os.listdir(xml_path)



    #if (os.path.exists(output_path)):
    #    os.removedirs(output_path)
    #os.makedirs(output_path)

    out_txts_path = os.path.join(output_path + '/' + 'labels')


    os.mkdir(out_txts_path)

    out_train_labels_path = os.path.join(out_txts_path + '/' + 'train')
    out_val_labels_path = os.path.join(out_txts_path + '/' + 'val')
    out_val_image_path=os.path.join(input_path+'/images/val')

    os.mkdir(out_val_image_path)
    os.mkdir(out_train_labels_path)
    os.mkdir(out_val_labels_path)

    random.seed(123)
    random.shuffle(xml_file_list)

    all_xml_cnt = len(xml_file_list)

    train_numb=int(all_xml_cnt*train_val_proportion)
    train_examples=xml_file_list[:train_numb]
    val_examples=xml_file_list[train_numb:]



    sum_dict = {}

    for xml_idx, per_xml in enumerate(val_examples):

        #if per_xml in exp_xml_list:
        #    continue
        if (xml_idx % 1000 == 0):
            print("validation calced: ", xml_idx, ' of ', len(val_examples))
        readTree = ET.ElementTree(file=xml_path + '/' + per_xml)
        root = readTree.getroot()  #

        filename = root.find("filename").text

        shutil.move(all_images_path +'/'+filename,out_val_image_path+'/'+filename)

        size = root.find("size")
        width = int(size.find("width").text)
        height = int(size.find("height").text)

        #img=cv2.imread(out_val_image_path+'/'+filename,0)
        #height,width=img.shape[0],img.shape[1]


        pure_filename = filename[:filename.rfind(".")]
        per_txt=open(out_val_labels_path+'/'+pure_filename+'.txt','w')
        for obj in root.findall('object'):#单张图片有多个目标
            bbox = obj.find('bndbox')
            if int(obj.find('name').text)<=13:
                class_idx=int(obj.find('name').text)-1#1~13 -> 0~12
            elif int(obj.find('name').text)>=16:
                class_idx = int(obj.find('name').text) - 2-1#16~19 -> 13~16
            #class_idx=int(obj.find('name').text)-1 #TODO:YOLOv5，Class numbers are zero-indexed (start from 0)
            if (sum_dict.get(class_idx) == None):
                sum_dict[class_idx] = 0
            else:
                sum_dict[class_idx] += 1

            # 获取bbox坐标信息
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            perline=str(class_idx)+' '+str(round((float)(xmin+xmax)/2.0/width,6))+' '+str(round((float)(ymin+ymax)/2.0/height,6))+' '
            perline+=str(round((float)(xmax-xmin)/width,6))+' '+str(round((float)(ymax-ymin)/height,6))
            per_txt.write(perline+'\n')
        per_txt.close()


    for xml_idx, per_xml in enumerate(train_examples):
        #if per_xml in exp_xml_list:
        #    continue
        if (xml_idx % 1000 == 0):
            print("train calced: ", xml_idx, ' of ', len(train_examples))
        readTree = ET.ElementTree(file=xml_path + '/' + per_xml)
        root = readTree.getroot()  #



        filename = root.find("filename").text

        size = root.find("size")
        width = int(size.find("width").text)
        height = int(size.find("height").text)

        #img = cv2.imread(all_images_path + '/' + filename, 0)
        #height, width = img.shape[0], img.shape[1]

        pure_filename = filename[:filename.rfind(".")]
        per_txt=open(out_train_labels_path+'/'+pure_filename+'.txt','w')
        for obj in root.findall('object'):#单张图片有多个目标
            bbox = obj.find('bndbox')
            if int(obj.find('name').text)<=13:
                class_idx=int(obj.find('name').text)-1#1~13 -> 0~12
            elif int(obj.find('name').text)>=16:
                class_idx = int(obj.find('name').text) - 2-1#16~19 -> 13~16
            #class_idx=int(obj.find('name').text)-1#Class numbers are zero-indexed (start from 0)
            if (sum_dict.get(class_idx) == None):
                sum_dict[class_idx] = 0
            else:
                sum_dict[class_idx] += 1

            # 获取bbox坐标信息
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            perline=str(class_idx)+' '+str(round((float)(xmin+xmax)/2.0/width,6))+' '+str(round((float)(ymin+ymax)/2.0/height,6))+' '
            perline+=str(round((float)(xmax-xmin)/width,6))+' '+str(round((float)(ymax-ymin)/height,6))
            per_txt.write(perline+'\n')
        per_txt.close()
    print(sum_dict)


def comb_poker_rect_and_convert_annotation_to_yolov5_data(input_path,output_path,train_val_proportion):
    annotations_path = os.path.join(input_path +'/'+ 'annotations')
    xml_path = os.path.join(annotations_path +'/'+ 'xmls')
    all_images_path = os.path.join(input_path +'/'+ 'images/train')

    xml_file_list = os.listdir(xml_path)



    #if (os.path.exists(output_path)):
    #    os.removedirs(output_path)
    #os.makedirs(output_path)

    out_txts_path = os.path.join(output_path + '/' + 'labels')


    os.mkdir(out_txts_path)

    out_train_labels_path = os.path.join(out_txts_path + '/' + 'train')
    out_val_labels_path = os.path.join(out_txts_path + '/' + 'val')
    out_val_image_path=os.path.join(input_path+'/images/val')

    os.mkdir(out_val_image_path)
    os.mkdir(out_train_labels_path)
    os.mkdir(out_val_labels_path)

    random.seed(123)
    random.shuffle(xml_file_list)

    all_xml_cnt = len(xml_file_list)

    train_numb=int(all_xml_cnt*train_val_proportion)
    train_examples=xml_file_list[:train_numb]
    val_examples=xml_file_list[train_numb:]



    sum_dict = {}
    aspect_dict={}

    for xml_idx, per_xml in enumerate(val_examples):

        #if per_xml in exp_xml_list:
        #    continue
        if (xml_idx % 1000 == 0):
            print("validation calced: ", xml_idx, ' of ', len(val_examples))
        readTree = ET.ElementTree(file=xml_path + '/' + per_xml)
        root = readTree.getroot()  #

        filename = root.find("filename").text

        shutil.move(all_images_path +'/'+filename,out_val_image_path+'/'+filename)

        size = root.find("size")
        width = int(size.find("width").text)
        height = int(size.find("height").text)

        #img=cv2.imread(out_val_image_path+'/'+filename,0)
        #height,width=img.shape[0],img.shape[1]


        pure_filename = filename[:filename.rfind(".")]
        per_txt=open(out_val_labels_path+'/'+pure_filename+'.txt','w')
        vec_rect_info=[]
        for obj in root.findall('object'):#单张图片有多个目标
            bbox = obj.find('bndbox')
            if int(obj.find('name').text)<=13:
                class_idx=int(obj.find('name').text)-1#1~13 -> 0~12
            elif int(obj.find('name').text)>=16:
                class_idx = int(obj.find('name').text) - 2-1#16~19 -> 13~16
            #class_idx=int(obj.find('name').text)-1 #TODO:YOLOv5，Class numbers are zero-indexed (start from 0)
            if (sum_dict.get(class_idx) == None):
                sum_dict[class_idx] = 0
            else:
                sum_dict[class_idx] += 1

            # 获取bbox坐标信息
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            vec_rect_info.append([xmin,ymin,xmax,ymax,class_idx])

        comb_rect=[width,height,0,0]
        for curr_rect in vec_rect_info:
            if curr_rect[0]<comb_rect[0]:
                comb_rect[0]=curr_rect[0]
            if curr_rect[1]<comb_rect[1]:
                comb_rect[1]=curr_rect[1]
            if curr_rect[2]>comb_rect[2]:
                comb_rect[2]=curr_rect[2]
            if curr_rect[3]>comb_rect[3]:
                comb_rect[3]=curr_rect[3]

        aspect_radio = float(comb_rect[2] - comb_rect[0]) / (comb_rect[3] - comb_rect[1])  # float(xmax-xmin)/(ymax-ymin)
        aspect_radio = round(aspect_radio, 1)
        if (aspect_dict.get(aspect_radio) == None):
            aspect_dict[aspect_radio] = [1, comb_rect[3] - comb_rect[1], comb_rect[2] - comb_rect[0]]
        else:
            aspect_dict[aspect_radio][0] += 1
            aspect_dict[aspect_radio][1] += (comb_rect[3] - comb_rect[1])
            aspect_dict[aspect_radio][2] += (comb_rect[2] - comb_rect[0])
        perline = str(0) + ' ' + str(round((float)(comb_rect[0] + comb_rect[2]) / 2.0 / width, 6)) + ' ' + str(round((float)(comb_rect[1] + comb_rect[3]) / 2.0 / height, 6)) + ' '
        perline += str(round((float)(comb_rect[2] - comb_rect[0]) / width, 6)) + ' ' + str(round((float)(comb_rect[3] - comb_rect[1]) / height, 6))
        per_txt.write(perline + '\n')
        per_txt.close()


    for xml_idx, per_xml in enumerate(train_examples):
        #if per_xml in exp_xml_list:
        #    continue
        if (xml_idx % 1000 == 0):
            print("train calced: ", xml_idx, ' of ', len(train_examples))
        readTree = ET.ElementTree(file=xml_path + '/' + per_xml)
        root = readTree.getroot()  #



        filename = root.find("filename").text

        size = root.find("size")
        width = int(size.find("width").text)
        height = int(size.find("height").text)

        #img = cv2.imread(all_images_path + '/' + filename, 0)
        #height, width = img.shape[0], img.shape[1]

        pure_filename = filename[:filename.rfind(".")]
        per_txt=open(out_train_labels_path+'/'+pure_filename+'.txt','w')
        vec_rect_info = []
        for obj in root.findall('object'):#单张图片有多个目标
            bbox = obj.find('bndbox')
            if int(obj.find('name').text)<=13:
                class_idx=int(obj.find('name').text)-1#1~13 -> 0~12
            elif int(obj.find('name').text)>=16:
                class_idx = int(obj.find('name').text) - 2-1#16~19 -> 13~16
            #class_idx=int(obj.find('name').text)-1#Class numbers are zero-indexed (start from 0)
            if (sum_dict.get(class_idx) == None):
                sum_dict[class_idx] = 0
            else:
                sum_dict[class_idx] += 1

            # 获取bbox坐标信息
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            vec_rect_info.append([xmin, ymin, xmax, ymax, class_idx])
        comb_rect = [width, height, 0, 0]
        for curr_rect in vec_rect_info:
            if curr_rect[0] < comb_rect[0]:
                comb_rect[0] = curr_rect[0]
            if curr_rect[1] < comb_rect[1]:
                comb_rect[1] = curr_rect[1]
            if curr_rect[2] > comb_rect[2]:
                comb_rect[2] = curr_rect[2]
            if curr_rect[3] > comb_rect[3]:
                comb_rect[3] = curr_rect[3]

        aspect_radio = float(comb_rect[2] - comb_rect[0]) / (
                    comb_rect[3] - comb_rect[1])  # float(xmax-xmin)/(ymax-ymin)
        aspect_radio = round(aspect_radio, 1)
        if (aspect_dict.get(aspect_radio) == None):
            aspect_dict[aspect_radio] = [1, comb_rect[3] - comb_rect[1], comb_rect[2] - comb_rect[0]]
        else:
            aspect_dict[aspect_radio][0] += 1
            aspect_dict[aspect_radio][1] += (comb_rect[3] - comb_rect[1])
            aspect_dict[aspect_radio][2] += (comb_rect[2] - comb_rect[0])
        perline = str(0) + ' ' + str(round((float)(comb_rect[0] + comb_rect[2]) / 2.0 / width, 6)) + ' ' + str(
            round((float)(comb_rect[1] + comb_rect[3]) / 2.0 / height, 6)) + ' '
        perline += str(round((float)(comb_rect[2] - comb_rect[0]) / width, 6)) + ' ' + str(
            round((float)(comb_rect[3] - comb_rect[1]) / height, 6))
        per_txt.write(perline + '\n')
        per_txt.close()

    print(sum_dict)

    for key in aspect_dict.keys():
        aspect_dict[key][1]=round(aspect_dict[key][1]/aspect_dict[key][0],1)
        aspect_dict[key][2] = round(aspect_dict[key][2] / aspect_dict[key][0], 1)
    for i in sorted(aspect_dict):
        print(i,aspect_dict[i])

def cover_part_obj(input_path, output_path):

    annotations_path = os.path.join(input_path + '/' + 'annotations')
    xml_path = os.path.join(annotations_path + '/' + 'xmls_all')
    images_path = os.path.join(input_path + '/' + 'images')
    xml_file = os.listdir(xml_path)
    trainvalTxt = open(annotations_path + '/' + 'no_cover.txt', 'w')
    if (os.path.exists(output_path)):
        os.removedirs(output_path)
    os.makedirs(output_path)

    out_annotations_path = os.path.join(output_path + '/' + 'annotations')
    os.mkdir(out_annotations_path)
    out_xml_path = os.path.join(out_annotations_path + '/' + 'xmls')
    os.mkdir(out_xml_path)
    out_images_path = os.path.join(output_path + '/' + 'images')
    os.mkdir(out_images_path)

    # aim_width,aim_height=680,680
    # print(aim_height,aim_width)
    # exp_xml_list=os.listdir('/home/tnc/PycharmProjects/DATA/object_detection/shuffle_poker/annotations/xmls')
    all_xml_cnt = len(xml_file)
    for xml_idx, per_xml in enumerate(xml_file):
        # if per_xml in exp_xml_list:
        #    continue
        if (xml_idx % 1000 == 0):
            print("calced: ", xml_idx, ' of ', all_xml_cnt)
        readTree = ET.ElementTree(file=xml_path + '/' + per_xml)
        root = readTree.getroot()  #

        size = root.find("size")
        width = int(size.find("width").text)
        height = int(size.find("height").text)
        filename = str(root.find("filename").text)
        #print(filename)
        per_img_rect = []
        mean_size = [0, 0]
        for obj in root.findall('object'):
            bbox = obj.find('bndbox')
            # 获取bbox坐标信息
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)

            # 修改bbox坐标信息
            # bbox.find('ymin').text = str(1)
            # bbox.find('ymax').text = str(2)
            # bbox.find('xmin').text = str(3)
            # bbox.find('xmax').text = str(4)
            # if (x1 == 262):
            #    root.remove(obj)
            per_img_rect.append([-1, xmin, ymin, xmax, ymax])
            mean_size[0] += xmax - xmin
            mean_size[1] += ymax - ymin
        mean_size[0] /= len(per_img_rect)
        mean_size[1] /= len(per_img_rect)
        #print(mean_size)
        if (len(per_img_rect) == 6):
            per_img_rect.sort(key=lambda x: x[1])
            if per_img_rect[0][2] > per_img_rect[1][2]:
                per_img_rect[0][0] = 3
                per_img_rect[1][0] = 0
            else:
                per_img_rect[0][0] = 0
                per_img_rect[1][0] = 3
            per_img_rect.sort(key=lambda x: x[3])
            if per_img_rect[4][2] > per_img_rect[5][2]:
                per_img_rect[4][0] = 5
                per_img_rect[5][0] = 2
            else:
                per_img_rect[4][0] = 2
                per_img_rect[5][0] = 5
            per_img_rect.sort(key=lambda x: x[0])
            col_val = [c[0] for c in per_img_rect]
            if col_val == [-1, -1, 0, 2, 3, 5]:
                if per_img_rect[0][2] > per_img_rect[1][2]:
                    per_img_rect[0][0] = 4
                    per_img_rect[1][0] = 1
                else:
                    per_img_rect[0][0] = 1
                    per_img_rect[1][0] = 4
                per_img_rect.sort(key=lambda x: x[0])
                #print(per_img_rect)
                image=cv2.imread(images_path+'/'+filename)
                #cv2.imshow  ("org",image)
                aim_per_img_rect=[]
                for rect in per_img_rect:
                    if rect[0]<3:#3,4,5
                        aim_per_img_rect.append(rect[1:])
                        #>=3 _0 下下打马，-1, xmin, ymin, xmax, ymax
                        #roi=image[rect[2]+int((rect[4]-rect[2])*0.33):rect[4],rect[1]:rect[3]]#y,x
                        #dst=cv2.GaussianBlur(roi, (0,0),8)
                        #image[rect[2]+int((rect[4]-rect[2])*0.33):rect[4],rect[1]:rect[3]]=dst

                        # <3 _1 上上打马,-1, xmin, ymin, xmax, ymax
                        roi = image[rect[2]:rect[2] + int((rect[4] - rect[2]) * 0.66), rect[1]:rect[3]]  # y,x
                        dst = cv2.GaussianBlur(roi, (0, 0), 8)
                        image[rect[2]:rect[2] + int((rect[4] - rect[2]) * 0.66), rect[1]:rect[3]] = dst

                #cv2.imshow("im", image)
                #cv2.waitKey(0)
                filename=filename[:filename.rfind('.')]+"_1"+filename[filename.rfind('.'):]
                #print(filename)
                cv2.imwrite(out_images_path + '/' + filename, image)

                for obj in root.findall('object'):
                    bbox = obj.find('bndbox')
                    # 获取bbox坐标信息
                    xmin = int(bbox.find('xmin').text)
                    ymin = int(bbox.find('ymin').text)
                    xmax = int(bbox.find('xmax').text)
                    ymax = int(bbox.find('ymax').text)
                    if [xmin,ymin,xmax,ymax] in aim_per_img_rect:
                        root.remove(obj)
                    # 修改bbox坐标信息
                    # bbox.find('ymin').text = str(1)
                    # bbox.find('ymax').text = str(2)
                    # bbox.find('xmin').text = str(3)
                    # bbox.find('xmax').text = str(4)

                per_xml=per_xml[:per_xml.rfind(".")]+"_1"+per_xml[per_xml.rfind("."):]
                root.find("filename").text=str(filename)
                readTree.write(out_xml_path + '/' + per_xml)
            else:
                trainvalTxt.write(per_xml + '\n')
        else:
            trainvalTxt.write(per_xml + '\n')


def SplitShuffleNBoImg(input_path,output_path_):
    annotations_path = os.path.join(input_path + '/' + 'annotations')
    xml_path = os.path.join(annotations_path + '/' + 'xmls')
    images_path = os.path.join(input_path + '/' + 'images')
    xml_file = os.listdir(xml_path)

    trainval_shuffle_Txt = open(annotations_path + '/' + 'trainval_XI.txt', 'a')
    trainval_bo_Txt = open(annotations_path + '/' + 'trainval_BO.txt', 'a')
    #for per_img in images:
    #    trainvalTxt.write(per_img[:per_img.find('.')] + '\n')  # jpg
    #trainvalTxt.close()


    all_xml_cnt = len(xml_file)
    for xml_idx, per_xml in enumerate(xml_file):
        # if per_xml in exp_xml_list:
        #    continue
        if (xml_idx % 1000 == 0):
            print("calced: ", xml_idx, ' of ', all_xml_cnt)
        readTree = ET.ElementTree(file=xml_path + '/' + per_xml)
        root = readTree.getroot()  #

        size = root.find("size")
        width = int(size.find("width").text)
        height = int(size.find("height").text)
        filename = str(root.find("filename").text)
        # print(filename)
        #per_img_rect = []
        box_cnt=0
        mean_size = [0, 0]
        for obj in root.findall('object'):
            bbox = obj.find('bndbox')
            # 获取bbox坐标信息
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)


            #per_img_rect.append([-1, xmin, ymin, xmax, ymax])
            mean_size[0] += xmax - xmin
            mean_size[1] += ymax - ymin
            box_cnt+=1
        mean_size[0] /= box_cnt
        mean_size[1] /= box_cnt

        aspect_radio=float(mean_size[0])/float(mean_size[1])

        if(aspect_radio>1.5):
            trainval_bo_Txt.write(filename[:filename.find('.')] + '\n')
        else:
            trainval_shuffle_Txt.write(filename[:filename.find('.')] + '\n')
    trainval_bo_Txt.close()
    trainval_shuffle_Txt.close()





def copy_part_file(input_path,output_path):
    annotations_path = os.path.join(input_path + '/' + 'annotations')
    xml_path = os.path.join(annotations_path + '/' + 'xmls')
    images_path = os.path.join(input_path + '/' + 'images')
    xml_file = os.listdir(xml_path)
    if (os.path.exists(output_path)):
        os.removedirs(output_path)
    os.makedirs(output_path)

    out_annotations_path = os.path.join(output_path + '/' + 'annotations')
    os.mkdir(out_annotations_path)
    out_xml_path = os.path.join(out_annotations_path + '/' + 'xmls')
    os.mkdir(out_xml_path)
    out_images_path = os.path.join(output_path + '/' + 'images')
    os.mkdir(out_images_path)

    all_flag_xml=[]
    all_cls_dist={}
    for idx,per_xml in enumerate(xml_file):
        if idx%1000==0:
            print("calc:",idx,"of", len(xml_file))
        readTree = ET.ElementTree(file=xml_path + '/' + per_xml)
        root = readTree.getroot()
        filename = str(root.find("filename").text)
        for obj in root.findall('object'):
            label = str(obj.find('name').text)
            all_flag_xml.append([int(label), per_xml,filename])
            if all_cls_dist.get(int(label))==None:
                all_cls_dist[int(label)]=0
            else:
                all_cls_dist[int(label)]+=1
            break
    all_flag_xml.sort(key=lambda x: x[0])
    per_aim_numb=300
    curr_cls_cnt =-1
    curr_cls=-1
    copy_dict={}
    for idx,per_xml in enumerate(all_flag_xml):
        if(curr_cls==-1):
            curr_cls=per_xml[0]
            curr_cls_cnt=all_cls_dist[curr_cls]
        else:
            if curr_cls!=per_xml[0]:
                curr_cls = per_xml[0]
                curr_cls_cnt = all_cls_dist[curr_cls]
        if idx%int(curr_cls_cnt/per_aim_numb)==0:
            if copy_dict.get(curr_cls)==None:
                copy_dict[curr_cls]=0
            else:
                copy_dict[curr_cls]+=1
            old_xml = xml_path + "/" + per_xml[1]
            new_xml = out_xml_path + "/" + per_xml[1]

            old_img = images_path + "/" + per_xml[2]
            new_img = out_images_path + "/" + per_xml[2]
            shutil.copyfile(old_xml, new_xml)
            shutil.copyfile(old_img, new_img)
    print(copy_dict)

def repair_2_square_data(input_path):
    annotations_path = os.path.join(input_path + '/' + 'annotations')
    xml_path = os.path.join(annotations_path + '/' + 'xmls')
    all_images_path = os.path.join(input_path + '/' + 'images/train')

    xml_file_list = os.listdir(xml_path)

    all_xml_cnt = len(xml_file_list)


    for xml_idx, per_xml in enumerate(xml_file_list):
        if (xml_idx % 1000 == 0):
            print("train calced: ", xml_idx, ' of ', all_xml_cnt)
        readTree = ET.ElementTree(file=xml_path + '/' + per_xml)
        root = readTree.getroot()  #

        size = root.find("size")
        width = int(size.find("width").text)
        height = int(size.find("height").text)
        filename = root.find("filename").text
        pure_filename = filename[:filename.rfind(".")]
        img_type=filename[filename.rfind("."):]
        if width!=height and os.path.exists(all_images_path+'/'+filename)==True:
            print(pure_filename)
            if width>height:
                curr_image=cv2.imread(all_images_path+'/'+filename,1)
                exp_img=cv2.copyMakeBorder(curr_image,0,width-height,0,0,cv2.BORDER_REFLECT)
                size.find("height").text=str(width)
            elif width<height:
                curr_image=cv2.imread(all_images_path+'/'+filename,1)
                exp_img=cv2.copyMakeBorder(curr_image,0,0,0,height-width,cv2.BORDER_REFLECT)
                size.find("width").text=str(height)
            os.remove(all_images_path+'/'+filename)
            cv2.imwrite(all_images_path+'/'+filename,exp_img)

            os.remove(xml_path + '/' + per_xml)
            readTree.write(xml_path + '/' +per_xml)


def writeLabelmeJson(input_path,output_path):
    annotations_path = os.path.join(input_path + '/' + 'annotations')
    xml_path = os.path.join(annotations_path + '/' + 'xmls')

    xml_file_list = os.listdir(xml_path)

    out_json_path = os.path.join(output_path + '/' + 'labelmeJson')
    os.mkdir(out_json_path)


    sum_dict = {}
    for xml_idx, per_xml in enumerate(xml_file_list):
        #if per_xml in exp_xml_list:
        #    continue
        if (xml_idx % 1000 == 0):
            print("train calced: ", xml_idx, ' of ', len(xml_file_list))
        readTree = ET.ElementTree(file=xml_path + '/' + per_xml)
        root = readTree.getroot()  #

        size = root.find("size")
        width = int(size.find("width").text)
        height = int(size.find("height").text)
        filename = root.find("filename").text
        pure_filename = filename[:filename.rfind(".")]
        #per_txt=open(out_train_labels_path+'/'+pure_filename+'.txt','w')


        record={}
        record['version'] = '5.0.1'
        record['flags']={}
        record['shapes'] = []

        instance = {
            #'line_color': None,
            #'fill_color': None,

        }

        for obj in root.findall('object'):#单张图片有多个目标
            bbox = obj.find('bndbox')
            polygon_dict = obj.find('polygon')
            class_idx=int(obj.find('name').text)-1#Class numbers are zero-indexed (start from 0)
            if (sum_dict.get(class_idx) == None):
                sum_dict[class_idx] = 0
            else:
                sum_dict[class_idx] += 1

            # 获取bbox坐标信息
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)

            tlX=float(polygon_dict.find('tlX').text)
            tlY=float(polygon_dict.find('tlY').text)
            trX=float(polygon_dict.find('trX').text)
            trY=float(polygon_dict.find('trY').text)
            blX=float(polygon_dict.find('blX').text)
            blY=float(polygon_dict.find('blY').text)
            brX=float(polygon_dict.find('brX').text)
            brY=float(polygon_dict.find('brY').text)

            #perline=str(class_idx)+' '+str(round((float)(xmin+xmax)/2.0/width,6))+' '+str(round((float)(ymin+ymax)/2.0/height,6))+' '
            #perline+=str(round((float)(xmax-xmin)/width,6))+' '+str(round((float)(ymax-ymin)/height,6))

            polygon=[]
            polygon.append([blX,blY])
            polygon.append([tlX, tlY])
            polygon.append([trX, trY])
            polygon.append([brX,brY])

            copy_instance = instance.copy()
            copy_instance.update({
                'label': str(class_idx),
                'points': polygon,
                'group_id': None,
                'shape_type': "polygon",
                'flags':{}
            })
            record['shapes'].append(copy_instance)

            record['imagePath'] = filename
            record['imageData']=None
            record['imageHeight']=height
            record['imageWidth']=width
            # fillColor = [255, 0, 0, 128]
            # lineColor = [0, 255, 0, 128]
            #record['lineColor']=fillColor
            #record['lineColor']=lineColor


            with open(os.path.join(out_json_path,pure_filename+'.json'), 'w') as jsonfile:
                json.dump(record, jsonfile, ensure_ascii=True, indent=2)
            #per_txt.write(perline+'\n')
        #per_txt.close()




def labelmeJson2PolygonYolov5Txt(input_path, output_path,train_val_proportion):
    all_images_path = os.path.join(input_path + '/' + 'images/train')

    label_json_path = os.path.join(input_path + '/' + 'labelmeJson')
    json_file_list = os.listdir(label_json_path)

    out_txt_path = os.path.join(output_path + '/' + 'label')
    os.mkdir(out_txt_path)

    out_train_json_path = os.path.join(out_txt_path + '/' + 'train')
    out_val_json_path = os.path.join(out_txt_path + '/' + 'val')
    os.mkdir(out_train_json_path)
    os.mkdir(out_val_json_path)

    out_val_image_path = os.path.join(input_path + '/images/val')
    os.mkdir(out_val_image_path)


    random.seed(123)
    random.shuffle(json_file_list)
    all_xml_cnt = len(json_file_list)

    train_numb = int(all_xml_cnt * train_val_proportion)
    train_examples = json_file_list[:train_numb]
    val_examples = json_file_list[train_numb:]

    keys = [
        "version",
        "imageData",
        "imagePath",
        "shapes",  # polygonal annotations
        "flags",  # image level flags
        "imageHeight",
        "imageWidth",
    ]
    shape_keys = [
        "label",
        "points",
        "group_id",
        "shape_type",
        "flags",
    ]

    sum_dict = {}

    for xml_idx, per_xml in enumerate(val_examples):

        # if per_xml in exp_xml_list:
        #    continue
        if (xml_idx % 1000 == 0):
            print("validation calced: ", xml_idx, ' of ', len(val_examples))

        with open(label_json_path + '/' + per_xml, "r") as f:
            data = json.load(f)
            # if data["imageData"] is not None:
            # imageData = base64.b64decode(data["imageData"])
            # if PY2 and QT4:
            #    imageData = utils.img_data_to_png_data(imageData)
            # else:
            # relative path from label file to relative path from cwd
            # imagePath = osp.join(osp.dirname(filename), data["imagePath"])
            # imageData = self.load_image_file(imagePath)
            flags = data.get("flags") or {}
            filename = data["imagePath"]
            height = float(data.get("imageHeight"))
            width = float(data.get("imageWidth"))

            shutil.move(all_images_path + '/' + filename, out_val_image_path + '/' + filename)

            pure_filename = filename[:filename.rfind(".")]
            per_txt = open(out_val_json_path + '/' + pure_filename + '.txt', 'w')

            shapes = [
                dict(
                    label=s["label"],
                    points=s["points"],
                    shape_type=s.get("shape_type", "polygon"),
                    flags=s.get("flags", {}),
                    group_id=s.get("group_id"),
                    other_data={
                        k: v for k, v in s.items() if k not in shape_keys
                    },
                )
                for s in data["shapes"]
            ]
            otherData = {}
            for key, value in data.items():
                if key not in keys:
                    otherData[key] = value
            for idx,lbNPo in enumerate(shapes):
                if (sum_dict.get(str(lbNPo['label'])) == None):
                    sum_dict[str(lbNPo['label'])] = 0
                else:
                    sum_dict[str(lbNPo['label'])] += 1
                label_idx=int(lbNPo['label']) #yoov5 idx from 0
                perline = str(label_idx)
                for pnt in lbNPo['points']:
                    perline+= ' ' + str(round(pnt[0]/width, 6)) + ' ' + str(round(pnt[1]/height, 6))
                per_txt.write(perline + '\n')
            per_txt.close()

    for xml_idx, per_xml in enumerate(train_examples):

        # if per_xml in exp_xml_list:
        #    continue
        if (xml_idx % 1000 == 0):
            print("train calced: ", xml_idx, ' of ', len(train_examples))

        with open(label_json_path + '/' + per_xml, "r") as f:
            data = json.load(f)
            # if data["imageData"] is not None:
            # imageData = base64.b64decode(data["imageData"])
            # if PY2 and QT4:
            #    imageData = utils.img_data_to_png_data(imageData)
            # else:
            # relative path from label file to relative path from cwd
            # imagePath = osp.join(osp.dirname(filename), data["imagePath"])
            # imageData = self.load_image_file(imagePath)
            flags = data.get("flags") or {}
            filename = data["imagePath"]
            height = float(data.get("imageHeight"))
            width = float(data.get("imageWidth"))

            #shutil.move(all_images_path + '/' + filename, out__image_path + '/' + filename)

            pure_filename = filename[:filename.rfind(".")]
            per_txt = open(out_train_json_path + '/' + pure_filename + '.txt', 'w')

            shapes = [
                dict(
                    label=s["label"],
                    points=s["points"],
                    shape_type=s.get("shape_type", "polygon"),
                    flags=s.get("flags", {}),
                    group_id=s.get("group_id"),
                    other_data={
                        k: v for k, v in s.items() if k not in shape_keys
                    },
                )
                for s in data["shapes"]
            ]
            otherData = {}
            for key, value in data.items():
                if key not in keys:
                    otherData[key] = value
            for idx,lbNPo in enumerate(shapes):
                if (sum_dict.get(str(lbNPo['label'])) == None):
                    sum_dict[str(lbNPo['label'])] = 0
                else:
                    sum_dict[str(lbNPo['label'])] += 1
                label_idx = int(lbNPo['label'])  # yoov5 idx from 0
                perline = str(label_idx)
                for pnt in lbNPo['points']:
                    perline+= ' ' + str(round(pnt[0]/width, 6)) + ' ' + str(round(pnt[1]/height, 6))
                per_txt.write(perline + '\n')
            per_txt.close()
    print("sum:",sum_dict)


def convert2Rectangle(src_img_path,out_img_path):
    os.mkdir(out_img_path)

    img_list=os.listdir(src_img_path)
    for idx, per_img_path in enumerate(img_list):
        #pure_img_path=per_img_path[0:per_img_path.find("/")]
        print(per_img_path)
        if idx%100==0:
            print("curr",idx,"of",len(img_list))
        per_image = cv2.imread(os.path.join(src_img_path,per_img_path), 0)
        src_height,src_width=per_image.shape[0],per_image.shape[1]
        add_x=(1280.0/720.0)*src_height-src_width
        mkborder_img=cv2.copyMakeBorder(per_image,0,0,0,int(add_x),cv2.BORDER_CONSTANT,value=(127,127,127))
        cv2.imwrite(os.path.join(out_img_path,per_img_path), mkborder_img)


def combImgNXml(type,src_path,out_path):
    if type==0:#image
        img_cnt=0
        os.mkdir(out_path)
        ft_list=os.listdir(src_path)
        for per_pure_obj_path in ft_list:
            per_whole_obj_path=os.path.join(src_path,per_pure_obj_path)
            image_list=os.listdir(per_whole_obj_path)
            print(per_pure_obj_path)
            for per_pure_img_path in image_list:
                per_whole_image_path=os.path.join(per_whole_obj_path,per_pure_img_path)
                img_type=per_pure_img_path[per_pure_img_path.find('.')+1:]
                if img_type=='jpg' or img_type=='jpeg':
                    out_img_path=os.path.join(out_path, per_pure_img_path)
                    print(per_whole_image_path,out_img_path)
                    img_cnt+=1
                    shutil.move(per_whole_image_path,out_img_path)
        print("image cnt:",img_cnt)
    elif type==1:
        xml_cnt = 0
        os.mkdir(out_path)
        ft_list = os.listdir(src_path)
        for per_pure_obj_path in ft_list:
            per_whole_obj_path = os.path.join(src_path, per_pure_obj_path)
            xmls_file_list = os.listdir(per_whole_obj_path)
            print(per_pure_obj_path)
            for per_xmls_file in xmls_file_list:
                if per_xmls_file =="xmls":
                    xmls_file_path=os.path.join(per_whole_obj_path,per_xmls_file)
                    xmls_list=os.listdir(xmls_file_path)
                    for per_pure_xml in xmls_list:
                        per_whole_xml=os.path.join(xmls_file_path,per_pure_xml)
                        per_out_xml_path=os.path.join(out_path,per_pure_xml)
                        shutil.move(per_whole_xml,per_out_xml_path)
                        xml_cnt+=1
                        print(per_pure_xml)
        print("xml cnt:", xml_cnt)














if __name__ =='__main__':

    #trainVal_path='/home/tnc/PycharmProjects/DATA/object_detection/tongzi_complete_input/'
    #out_path='/home/tnc/PycharmProjects/DATA/object_detection/tongzi_complete_output_test/'

    #resize_scale=1
    #iter_obj(trainVal_path,out_path)
    #create_object_detection_sample(trainVal_path,out_path,resize_scale)
    #modify_image()

    #change_xml_cls('/media/tnc/BA62CCAF62CC71A5/object_detection/shuffle_poker_out/annotations/e_xmls',
    #               '/media/tnc/BA62CCAF62CC71A5/object_detection/shuffle_poker_out/annotations/xml24_out')
    #todo:生成trainval.txt
    #create_trainval_txt_and_calc_aspect_radio(
    #    '/media/tnc/BA62CCAF62CC71A5/object_detection/tongzi','')

    #tmp_xml('/home/tnc/PycharmProjects/DATA/object_detection/shuffle_poker/annotations/xmls','/home/tnc/PycharmProjects/DATA/object_detection/shuffle_poker/annotations/xmll')


    #rename_xml_filename('/home/tnc/PycharmProjects/DATA/object_detection/shuffle_poker/annotations/xmls',
    #                    '/home/tnc/PycharmProjects/DATA/object_detection/shuffle_poker/annotations/xmls1')

    #get_aim_img('/home/tnc/PycharmProjects/DATA/object_detection/shuffle_poker',
    #            '/home/tnc/PycharmProjects/DATA/object_detection/sf_t4')


    #todo:生成小图查验
    #cheak_worng_img_n_label('/home/tnc/PycharmProjects/DATA/object_detection/poker/images/train',
    #                        0,
    #                        '/home/tnc/PycharmProjects/DATA/object_detection/poker/annotations/xmls',
    #                        '',
    #                        '/home/tnc/PycharmProjects/DATA/object_detection/poker/out',20)

    #cheak_worng_img_n_label('/media/tnc/BA62CCAF62CC71A5/object_detection/shuffle_poker_out/images',
    #                        1,
    #                        '/media/tnc/BA62CCAF62CC71A5/object_detection/shuffle_poker_out/annotations/xmls',
    #                        '/media/tnc/BA62CCAF62CC71A5/object_detection/shuffle_poker_out/annotations/trainval_BO.txt',
    #                        '/media/tnc/BA62CCAF62CC71A5/object_detection/shuffle_poker_out/out_bo',52)

    #todo:数据数据分析
    #calc_aspect_radio_n_size('/home/tnc/PycharmProjects/DATA/object_detection/poker','')



    #todo:扩大图片
    #expand_img_n_change_xml('/media/tnc/BA62CCAF62CC71A5/object_detection/shuffle_poker',
    #                        '/media/tnc/BA62CCAF62CC71A5/object_detection/shuffle_poker_out')


    #cover_part_obj('/media/tnc/BA62CCAF62CC71A5/object_detection/tongzi_same_6',
    #               '/media/tnc/BA62CCAF62CC71A5/object_detection/tongzi_same6_0610')

    #copy_part_file('/media/tnc/BA62CCAF62CC71A5/object_detection/tongzi_same_6',
    #               '/media/tnc/BA62CCAF62CC71A5/object_detection/tongzi_same6_0610')

    #SplitShuffleNBoImg("/media/tnc/BA62CCAF62CC71A5/object_detection/shuffle_poker_out","")


    #repair_2_square_data('/media/tnc/6A68482A6847F37D/object_detection/tongzi')
    #todo:yolov5
    #convert_annotation_to_yolov5_data('/home/tnc/PycharmProjects/DATA/object_detection/poker_rect','/home/tnc/PycharmProjects/DATA/object_detection/poker_rect',0.96)

    #todo:转换贴灰图的poker样本
    comb_poker_rect_and_convert_annotation_to_yolov5_data('/home/tnc/PycharmProjects/DATA/object_detection/dealer_rect','/home/tnc/PycharmProjects/DATA/object_detection/dealer_rect',0.96)

    #todo:把非同凡响objdt转换到labelme Json
    #writeLabelmeJson('/home/wegn/PycharmProjects/DATA/paigow_long_polygon','/home/wegn/PycharmProjects/DATA/paigow_long_polygon')

    #todo:把labelme json文件转换到yolov5 polygon txt 文件
    #labelmeJson2PolygonYolov5Txt('/home/tnc/PycharmProjects/DATA/paigow_long_polygon','/home/tnc/PycharmProjects/DATA/paigow_long_polygon',0.92)

    #todo:转换成长方形图片
    #convert2Rectangle('/home/tnc/PycharmProjects/DATA/object_detection/poker/images/val','/home/tnc/PycharmProjects/DATA/object_detection/poker/images/val_out')

    #combImgNXml(1,"/home/tnc/PycharmProjects/DATA/object_detection/dealer_rect/xml_org",'/home/tnc/PycharmProjects/DATA/object_detection/dealer_rect/xml')