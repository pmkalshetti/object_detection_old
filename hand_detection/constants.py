import tensorflow.contrib.eager as tfe
import numpy as np
import os

# data
OBJECTS = ['hand']
NUM_OBJECTS = 1
MAX_DETECTIONS_PER_IMAGE = 1
VIDEO_IN = 'video.mp4'

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

# prediction
CHECKPOINT_DIR = 'model'
CHECKPOINT_PREFIX = os.path.join(CHECKPOINT_DIR, "ckpt")
DIR_IMGS_OUT =  'imgs_out'
THRESHOLD_OUT_PROB = 0.5
THRESHOLD_IOU_NMS = 0.5
if tfe.num_gpus() > 0:
    DEVICE = '/gpu:0'
    print('Using GPU')
else:
    DEVICE = '/cpu:0'
    print('Using CPU')
