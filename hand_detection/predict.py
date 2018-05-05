import tensorflow as tf
import tensorflow.contrib.eager as tfe
tf.enable_eager_execution(device_policy=tfe.DEVICE_PLACEMENT_SILENT)
from model import Model
from constants import *
from utils import *
import cv2
import os

# load model
with tf.device(DEVICE):
    model = Model()
    checkpoint = tfe.Checkpoint(model=model, optimizer_step=tf.train.get_or_create_global_step())
    checkpoint.restore(tf.train.latest_checkpoint(CHECKPOINT_DIR))
    print('Model loaded')
    
# perform prediction
if not os.path.exists(DIR_IMGS_OUT):
    os.makedirs(DIR_IMGS_OUT)
    
# check if webcam or video
WEBCAM =  False
if WEBCAM:
    video = 0
else:
    video = VIDEO_IN

font = cv2.FONT_HERSHEY_SIMPLEX
with tf.device(DEVICE):
    video_capture = cv2.VideoCapture(video)
    idx_img = 0
    while video_capture.isOpened():
        # read image
        ret, img = video_capture.read()
        
        if ret==True:
            img = process_img(img)
            
            # predict
            output = model.predict(img)
            
            # write images
            img_out = draw_output(img[0], output[0].numpy())
            cv2.imshow('prediction', img_out)
            #cv2.imwrite(DIR_IMGS_OUT+ '/'+str(idx_img) + '.jpg', img_out)
            
            if cv2.waitKey(1) & 0xFF == ord('\x1b'):
                break
        else:
            break
         
        idx_img += 1
        

