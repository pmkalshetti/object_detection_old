import tensorflow.contrib.eager as tfe
import numpy as np
import os

# VOC data
DIR_DATA = '../data'
DIR_INPUT = os.path.join(DIR_DATA, 'input')
DIR_OUTPUT = os.path.join(DIR_DATA, 'annotations')
DIR_TEST = os.path.join(DIR_DATA, 'test_input')
OBJECT_LABELS = {
    'tvmonitor': (0, 'Indoor'),
    'aeroplane': (1, 'Vehicle'),
    'bicycle': (2, 'Vehicle'),
    'bird': (3, 'Animal'),
    'boat': (4, 'Vehicle'),
    'bottle': (5, 'Indoor'),
    'bus': (6, 'Vehicle'),
    'car': (7, 'Vehicle'),
    'cat': (8, 'Animal'),
    'chair': (9, 'Indoor'),
    'cow': (10, 'Animal'),
    'diningtable': (11, 'Indoor'),
    'dog': (12, 'Animal'),
    'horse': (13, 'Animal'),
    'motorbike': (14, 'Vehicle'),
    'person': (15, 'Person'),
    'pottedplant': (16, 'Indoor'),
    'sheep': (17, 'Animal'),
    'sofa': (18, 'Indoor'),
    'train': (19, 'Vehicle')
}
OBJECTS = ['tvmonitor', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 
          'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train']
NUM_OBJECTS = 20
MAX_DETECTIONS_PER_IMAGE = 10

# data preprocessing
DIM_OUTPUT_PER_GRID_PER_ANCHOR = 5 + NUM_OBJECTS
GRID_H, GRID_W = 13, 13 
GRID_SIZE = 416//GRID_H
ANCHORS = np.array(
    [
        [0.09112895, 0.06958421],
        [0.21102316, 0.16803947],
        [0.42625895, 0.26609842],
        [0.25476474, 0.49848   ],
        [0.52668947, 0.59138947]
    ]
)
NUM_ANCHORS = ANCHORS.shape[0]
ANCHORS *= np.array([GRID_H, GRID_W])  # map from [0,1] space to [0,19] space
IMG_OUT_H, IMG_OUT_W = GRID_H * GRID_SIZE, GRID_W * GRID_SIZE 
DIR_TFRECORDS = os.path.join(DIR_DATA, 'data_tfrecords')
NUM_EXAMPLES_PER_TFRECORD = 10

# training
THRESHOLD_IOU_SCORES = 0.6
COEFF_LOSS_CONFIDENCE_OBJECT_PRESENT = 5
COEFF_LOSS_CONFIDENCE_OBJECT_ABSENT = 1
THRESHOLD_OUT_PROB = 0.5
THRESHOLD_IOU_NMS = 0.5
NUM_EPOCHS = 2
BATCH_SIZE = 16
CHECKPOINT_DIR = '../model'
CHECKPOINT_PREFIX = os.path.join(CHECKPOINT_DIR, "ckpt")

# prediction
VIDEO_IN = 'video.mp4'
DIR_IMGS_OUT =os.path.join('.',  'test_output')
if tfe.num_gpus() > 0:
    DEVICE = '/gpu:0'
    print('Using GPU')
else:
    DEVICE = '/cpu:0'
    print('Using CPU')
