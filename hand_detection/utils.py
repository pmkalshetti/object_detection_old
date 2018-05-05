import tensorflow as tf
from constants import *
import numpy as np
import cv2

'''
Functions required for model
'''
def apply_transformations(predictions):
    predictions_yx = tf.sigmoid(predictions[..., 0:2])
    predictions_hw = tf.exp(predictions[...,2:4])
    predictions_prob_obj = tf.sigmoid(predictions[...,4:5])
    predictions_prob_class = tf.nn.softmax(predictions[...,5:])
    
    return predictions_yx, predictions_hw, predictions_prob_obj, predictions_prob_class

def get_coordinates(h, w):
    coordinates_y = tf.range(h)
    coordinates_x = tf.range(w)
    x, y = tf.meshgrid(coordinates_x, coordinates_y)
    coordinates = tf.stack([y, x], axis=-1)
    coordinates = tf.reshape(coordinates, [1, h, w, 1, 2])
    coordinates = tf.cast(coordinates, tf.float32)
    
    return coordinates

def grid2normalized(predictions_yx, predictions_hw):    
    # create cartesian coordinates on grid space
    coordinates = get_coordinates(GRID_H, GRID_W)
    
    # map from grid space to [0,19] space
    anchors = tf.cast(tf.reshape(ANCHORS, [1, 1, 1, ANCHORS.shape[0], 2]), dtype=tf.float32)  # [0,19] space
    predictions_yx += coordinates
    predictions_hw *= anchors
    
    # map from [0,19] space to [0,1] space
    shape = tf.cast(tf.reshape([GRID_H, GRID_W], [1, 1, 1, 1, 2]), tf.float32)
    predictions_yx /= shape
    predictions_hw /= shape
    
    return predictions_yx, predictions_hw

def get_boxes_gt(args_map):
    # extract ground truth bboxes wherever prob_obj = 1
    mask_object = tf.cast(tf.reshape(args_map[1], [GRID_H, GRID_W, NUM_ANCHORS]), tf.bool)
    bboxes = tf.boolean_mask(args_map[0], mask_object)
    # bboxes.shape = [NUM_DETECTIONS, 4]; NUM_DETECTIONS vary with each image
    
    # pad bboxes so that bboxes is fixed dimension (fix NUM_DETECTIONS to MAX_DETECTIONS_PER_IMAGE)
    pad = tf.zeros((MAX_DETECTIONS_PER_IMAGE - tf.shape(bboxes)[0], 4))  # TODO: when NUM_DETECTIONS > MAX_DETECTIONS_PER_IMAGE
    bboxes = tf.concat([bboxes, pad], axis=0)
    
    return bboxes

def get_iou_scores(predictions_yx, predictions_hw, bboxes_gt):
    # predictions_yx.shape = predictions_hw.shape = [BATCH_SIZE, GRID_H, GRID_W, NUM_ANCHORS, 2]
    # bboxes_gt.shape = [BATCH_SIZE, MAX_DETECTIONS_PER_IMAGE, 4]
    
    # compute ious for each anchor in each grid in axis=4
    predictions_yx = tf.expand_dims(predictions_yx, 4)
    predictions_hw = tf.expand_dims(predictions_hw, 4)
    
    predictions_min = predictions_yx - predictions_hw/2.
    predictions_max = predictions_yx + predictions_hw/2.
    
    bboxes_gt = tf.reshape(bboxes_gt, [tf.shape(bboxes_gt)[0], 1, 1, 1, MAX_DETECTIONS_PER_IMAGE, 4])
    bboxes_gt_yx = bboxes_gt[..., 0:2]
    bboxes_gt_hw = bboxes_gt[..., 2:4]
    
    bboxes_gt_min = bboxes_gt_yx - bboxes_gt_hw/2.
    bboxes_gt_max = bboxes_gt_yx + bboxes_gt_hw/2.
    
    intersection_min = tf.maximum(predictions_min, bboxes_gt_min)
    intersection_max = tf.minimum(predictions_max, bboxes_gt_max)
    intersection_hw = tf.maximum(intersection_max - intersection_min, 0.)
    area_intersection = intersection_hw[..., 0] * intersection_hw[..., 1]
    
    area_predictions = predictions_hw[...,0] * predictions_hw[...,1]
    area_bboxes_gt = bboxes_gt_hw[...,0] * bboxes_gt_hw[...,1]
    area_union = area_bboxes_gt + area_predictions - area_intersection
    iou = area_intersection / area_union
    
    return iou


'''
Functions required for prediction
'''
def process_img(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_OUT_W, IMG_OUT_H))
    img = (img / 255.).astype(np.float32)
    img = np.expand_dims(img, 0)
    
    return img

def draw_output(img, output):
    # unnormalize image
    img = (img * 255).astype(np.uint8)
    
    for idx_box in range(output.shape[0]):
        conf = output[idx_box][4]
        bbox = output[idx_box].astype(np.int32)
        obj_class = OBJECTS[bbox[5]]
        img = cv2.rectangle(img, (bbox[1], bbox[0]), (bbox[3], bbox[2]), color=(255, 0, 0), thickness=3)
        img = cv2.line(img, (bbox[1]+10, bbox[0]+12), (bbox[1]+10+150, bbox[0]+12), (255, 0, 0), 20, cv2.LINE_AA)
        font = cv2.FONT_HERSHEY_SIMPLEX
        img = cv2.putText(img, '{} ({:.2f})'.format(obj_class, conf),(bbox[1]+5, bbox[0]+15), font, .5,(255,255,255),1,cv2.LINE_AA)
        
        
    return img

def center2corner(predictions_yx, predictions_hw):
    # predictions_yx = [GRID_H, GRID_W, NUM_ANCHORS, 2]
    
    bbox_min = predictions_yx - (predictions_hw/2.)
    bbox_max = predictions_yx + (predictions_hw/2.)
    
    predictions_corner = tf.concat([bbox_min[...,0:1], bbox_min[...,1:2], bbox_max[...,0:1], bbox_max[...,1:2]], axis=-1)
    
    return predictions_corner

def get_filtered_predictions(predictions_corner, predictions_prob_obj, predictions_prob_class):
    # compute overall prob for each anchor in each grid
    predictions_prob = predictions_prob_obj * predictions_prob_class
    
    # get max prob among all classes at each anchor in each grid
    predictions_idx_class_max = tf.argmax(predictions_prob, axis=-1)
    predictions_prob = tf.reduce_max(predictions_prob, axis=-1)
    
    # compute filter mask
    mask_filter = predictions_prob >= THRESHOLD_OUT_PROB
    
    # apply mask on output
    bbox_filtered = tf.boolean_mask(predictions_corner, mask_filter)
    prob_filtered = tf.boolean_mask(predictions_prob, mask_filter)
    with tf.device('/cpu:0'):
        idx_class_filtered = tf.boolean_mask(predictions_idx_class_max, mask_filter)
    
    return bbox_filtered, prob_filtered, idx_class_filtered


def predictions2outputs(predictions):
    # apply corresponding transformations on predictions
    predictions_yx, predictions_hw, predictions_prob_obj, predictions_prob_class = apply_transformations(predictions)
    
    # map predictions_bbox to [0,1] space
    predictions_yx, predictions_hw = grid2normalized(predictions_yx, predictions_hw)
    
    # represent boxes using corners
    predictions_corner = center2corner(predictions_yx, predictions_hw)
    
    # filter predictions based on (prob_obj * prob_class). (needs to be done separately for each image in batch)
    bbox_filtered, prob_filtered, idx_class_filtered = get_filtered_predictions(predictions_corner, predictions_prob_obj, predictions_prob_class)
    # bbox_filtered.shape = [BATCH_SIZE, NUM_FILTERED, 4]
    
    # TODO: perform nms for each class separately
    # scale boxes from [0,1] to image space
    img_space = tf.reshape(tf.cast(tf.stack([IMG_OUT_H, IMG_OUT_W, IMG_OUT_H, IMG_OUT_W]), tf.float32), [1, 1, 4])
    bbox_filtered = tf.reshape(bbox_filtered*img_space, [-1, 4])  # tf.nms takes num_boxes (no batch support)
    
    # perform non-max suppression
    # perform non-max suppression
    with tf.device('/cpu:0'):
        bbox_nms_indices = tf.image.non_max_suppression(bbox_filtered, tf.reshape(prob_filtered,[-1]), MAX_DETECTIONS_PER_IMAGE, THRESHOLD_IOU_NMS)
    if DEVICE == '/gpu:0':
        bbox_nms_indices = bbox_nms_indices.gpu()
    
    bbox_nms = tf.gather(bbox_filtered, bbox_nms_indices)  # box_nms.shape = [len(bbox_nms_indices), 4]
    prob_nms = tf.expand_dims(tf.gather(prob_filtered, bbox_nms_indices), axis=-1) # prob_nms.shape = [len(bbox_nms_indices), 1]
    with tf.device('/cpu:0'):
        idx_class_nms = tf.expand_dims(tf.cast(tf.gather(idx_class_filtered, bbox_nms_indices), tf.float32), axis=-1)
    if DEVICE == '/gpu:0':
        idx_class_nms = idx_class_nms.gpu()
    
    # concat return data
    output = tf.concat([bbox_nms, prob_nms, idx_class_nms], axis=-1)

    return tf.expand_dims(output, axis=0)

