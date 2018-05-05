from constants import *
import os
from glob import glob
import tensorflow as tf
import cv2
import xml.etree.ElementTree as ET
from math import floor

def read_data(filename):
    # Reference: This function has been modified from 
    # https://github.com/balancap/SSD-Tensorflow/blob/master/datasets/pascalvoc_to_tfrecords.py
    
    # read and process image
    img_name = os.path.join(DIR_INPUT, filename + '.jpg')
    img = cv2.imread(img_name)
    img_in_h = img.shape[0]
    img_in_w = img.shape[1]
    img = cv2.resize(img, (IMG_OUT_W, IMG_OUT_H))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # read annotation
    annotation_name = os.path.join(DIR_OUTPUT, filename + '.xml')
    tree = ET.parse(annotation_name)
    root = tree.getroot()
    targets = []
    for obj in root.findall('object'):
        # read class label
        label_text = obj.find('name').text
        label = int(OBJECT_LABELS[label_text][0])
        
        # read bbox
        bbox = obj.find('bndbox')
        y_min = float(bbox.find('ymin').text)
        x_min = float(bbox.find('xmin').text)
        y_max = float(bbox.find('ymax').text)
        x_max = float(bbox.find('xmax').text)
        
        # convert from corner coordinates to x_center, y_center, width, height
        y_center, x_center = (y_min + y_max)/2., (x_min + x_max)/2.
        bbox_h, bbox_w = y_max - y_min, x_max - x_min
        
        # normalize these values s.t. image goes from 0 to 1 (helps for arbitary size image size)
        y_center /= img_in_h
        x_center /= img_in_w
        bbox_h /= img_in_h
        bbox_w /= img_in_w

        targets.append((y_center, x_center, bbox_h, bbox_w, label))
        
    return img, np.array(targets, dtype=np.float32)

def get_iou(hw1, hw2):
    # hw: (height, width)
    # assumption: both boxes have same centers
    
    # get extremes of both boxes
    hw1_max, hw2_max = hw1/2., hw2/2.
    hw1_min, hw2_min = -hw1_max, -hw2_max
    
    # get intersection area
    intersection_min = np.maximum(hw1_min, hw2_min)
    intersection_max = np.minimum(hw1_max, hw2_max)
    hw_intersection = np.maximum(intersection_max-intersection_min, 0.)
    area_intersection = hw_intersection[0] * hw_intersection[1]
    
    # get union area
    area_hw1 = hw1[0] * hw1[1]
    area_hw2 = hw2[0] * hw2[1]
    area_union = area_hw1 + area_hw2 - area_intersection
    
    iou = area_intersection / area_union
    
    return iou

def targets2label(targets):
    # initialize return data
    label = np.zeros((GRID_H, GRID_W, NUM_ANCHORS, 6), dtype=np.float32)  # 6: [offset_y, offset_x, scale_h, scale_w, class_idx, prob_obj]
    
    # check for all targets
    for target in targets:
        target_class = target[4]
        
        # map bbox from [0,1] space to [0,13] space
        bbox = target[0:4] * np.array([GRID_H, GRID_W, GRID_H, GRID_W])
        
        # get grid index for bbox center
        idx_y = int(floor(bbox[0]))
        idx_x = int(floor(bbox[1]))
        
        # find best anchor corresponding to bbox
        iou_best, idx_anchor_best = 0., 0
        for idx_anchor, anchor in enumerate(ANCHORS):
            iou = get_iou(bbox[2:4], anchor)
            if iou > iou_best:
                iou_best = iou
                idx_anchor_best = idx_anchor
            
        # update label
        if iou_best > 0.:
            label[idx_y, idx_x, idx_anchor_best] = np.array(
                [
                    bbox[0] - idx_y,  # offset of box_center from top-left corner of grid containing box_center
                    bbox[1] - idx_x,
                    bbox[2]/ANCHORS[idx_anchor_best,0], # scale of anchor box so as to fit the bbox
                    bbox[3]/ANCHORS[idx_anchor_best,1],
                    target_class,
                    1.0  # prob_object (object is present with prob=1)
                ], dtype=np.float32
            )
    return label
        
def get_processed_data(filename):
    # read input and output
    img, targets = read_data(filename)
    # targets.shape = (num_objects, 5)
    # 5 corresponds to (c_y, c_x, h, w, class_label)
    
    label = targets2label(targets)
    
    return img, label
    
# conversion functions (data to feature data types)
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def write_example_to_TFRecord(filename, writer):
    # get processed data
    img, label = get_processed_data(filename)
    # create example from this data
    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                'img': _bytes_feature(img.tostring()),
                'label': _bytes_feature(label.tostring()),
            }
        )
    )

    writer.write(example.SerializeToString())


def write_data_to_TFRecord():        
    # read filenames
    filenames = sorted(os.listdir(DIR_INPUT))
    filenames = [filename[:-4] for filename in filenames]  # trim extension    
    
    # write data into multiple TFRecord files
    idx_tfrecord, idx_data = 0, 0
    if not os.path.exists(DIR_TFRECORDS):
        os.makedirs(DIR_TFRECORDS)
    
    while idx_data < len(filenames):
        # new TFRecord file
        filename_tfrecord = os.path.join(DIR_TFRECORDS, str(idx_tfrecord) + '.tfrecords')
        with tf.python_io.TFRecordWriter(filename_tfrecord) as writer:
            # write examples into this file until limit is reached
            idx_example = 0
            while idx_data < len(filenames) and idx_example < NUM_EXAMPLES_PER_TFRECORD:
                filename = filenames[idx_data]
                write_example_to_TFRecord(filename, writer)
                idx_data += 1
                idx_example += 1
            idx_tfrecord += 1

if __name__ == '__main__':
    write_data_to_TFRecord()
