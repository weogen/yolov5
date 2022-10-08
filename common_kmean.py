import numpy as np
import json
import os
from PIL import Image


def iou(box, clusters):
    """
   Calculate IOU
    param:
        box: tuple or array, shifted to the origin (i. e. width and height)
        clusters: numpy array of shape (k, 2) where k is the number of clusters
    return:
        numpy array of shape (k, 0) where k is the number of clusters
    """
    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])
    if np.count_nonzero(x == 0) > 0 or np.count_nonzero(y == 0) > 0:
        raise ValueError("Box has no area")

    intersection = x * y
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]

    iou_ = intersection / (box_area + cluster_area - intersection + 1e-10)

    return iou_


# Calculate the average and set intersection between the Numpy arrays and K clusters (IOU).
def avg_iou(boxes, clusters):
    """
    param:
        boxes: numpy array of shape (r, 2), where r is the number of rows
        clusters: numpy array of shape (k, 2) where k is the number of clusters
    return:
        average IoU as a single float
    """
    return np.mean([np.max(iou(boxes[i], clusters)) for i in range(boxes.shape[0])])


# Convert all boxes to the origin.
def translate_boxes(boxes):
    """
    param:
        boxes: numpy array of shape (r, 4)
    return:
    numpy array of shape (r, 2)
    """
    new_boxes = boxes.copy()
    for row in range(new_boxes.shape[0]):
        new_boxes[row][2] = np.abs(new_boxes[row][2] - new_boxes[row][0])
        new_boxes[row][3] = np.abs(new_boxes[row][3] - new_boxes[row][1])
    return np.delete(new_boxes, [0, 1], axis=1)


# Use the joint intersection (IOU) metric to calculate the K mean cluster.
def kmeans(boxes, k, dist=np.median):
    """
    param:
        boxes: numpy array of shape (r, 2), where r is the number of rows
        k: number of clusters
        dist: distance function
    return:
        numpy array of shape (k, 2)
    """
    rows = boxes.shape[0]

    distances = np.empty((rows, k))
    last_clusters = np.zeros((rows,))

    np.random.seed()

    # the Forgy method will fail if the whole array contains the same rows
    Clusters = Boxes[np.random.choice(ROWS, K,
                                      REPLACE=false]  # Initializing K poly class (method is randomly selected from the original data set)


while True:
    for row in range(rows):
        # Distance metric formula: D (Box, Centroid) = 1-IOU (Box, Centroid). The smaller the distance from the center of the cluster, the bigger the IOU value, so use 1 - IOU, so that the smaller the distance, the greater the IOU value.
        distances[row] = 1 - iou(boxes[row], clusters)
        # Assign the label box to the nearest cluster center (that is, the code is to select (for each box) to the cluster center).
    nearest_clusters = np.argmin(distances, axis=1)
    # Until the cluster center changes to 0 (that is, the cluster center is unchanged).
    if (last_clusters == nearest_clusters).all():
        break
        # Update Cluster Center (here the median mediterraneous number of classes as new clustering center)
    for cluster in range(k):
        clusters[cluster] = dist(boxes[nearest_clusters == cluster], axis=0)

    last_clusters = nearest_clusters

return clusters


#
def get_image_width_high(full_image_name):
    image = Image.open(full_image_name)
    image_width, image_high = image.size[0], image.size[1]
    return image_width, image_high


# Read the annotation data in the JSON file
def parse_label_json(label_path):
    with open(label_path, 'r') as f:
        label = json.load(f)
    result = []
    for line in label:
        bbox = line['bbox']
        x_label_min, y_label_min, x_label_max, y_label_max = bbox[0], bbox[1], bbox[2], bbox[3]
        # Calculate the size of the border
        width = x_label_max - x_label_min
        height = y_label_max - y_label_min
        assert width > 0
        assert height > 0
        result.append([width, height])
    result = np.asarray(result)
    return result


#     t
def parse_label_txt(label_path):
    all_label = os.listdir(label_path)
    result = []
    for i in range(len(all_label)):
        full_label_name = os.path.join(label_path, all_label[i])
        print(full_label_name)
        #   File name and file suffix
        label_name, label_extension = os.path.splitext(all_label[i])
        full_image_name = os.path.join(label_path.replace('labels', 'images'), label_name + '.jpg')
        image_width, image_high = get_image_width_high(full_image_name)
        fp = open(full_label_name, mode="r")
        lines = fp.readlines()
        for line in lines:
            array = line.split()
            x_label_min = (float(array[1]) - float(array[3]) / 2) * image_width
            x_label_max = (float(array[1]) + float(array[3]) / 2) * image_width
            y_label_min = (float(array[2]) - float(array[4]) / 2) * image_high
            y_label_max = (float(array[2]) + float(array[4]) / 2) * image_high
            # Calculate the size of the border
            width = x_label_max - x_label_min
            height = y_label_max - y_label_min
            assert width > 0
            assert height > 0
            result.append([round(width, 2), round(height, 2)])
    result = np.asarray(result)

    return result


def get_kmeans(label, cluster_num=9):
    anchors = kmeans(label, cluster_num)
    ave_iou = avg_iou(label, anchors)

    anchors = anchors.astype('int').tolist()

    anchors = sorted(anchors, key=lambda x: x[0] * x[1])

    return anchors, ave_iou


if __name__ == '__main__':
    #   j j
    label_path = "tile_round1_train_20201231/train_annos.json"
    label_result = parse_label_json(label_path)

    #
    # label_path = "../Image_data/seed/labels/" # seed / images / is the corresponding image file
    # label_result = parse_label_txt(label_path)

    anchors, ave_iou = get_kmeans(label_result, 9)

    anchor_string = ''
    for anchor in anchors:
        anchor_string += '{},{}, '.format(anchor[0], anchor[1])
    anchor_string = anchor_string[:-2]

    print(f'anchors are: {anchor_string}')
    print(f'the average iou is: {ave_iou}')