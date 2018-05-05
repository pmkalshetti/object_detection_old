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

MODE = 1  # webcam:0, video:1, imgs:2
if MODE == 0:
    video = 0
elif MODE == 1:
    video = VIDEO_IN

if MODE == 0 or MODE == 1:
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
              #cv2.imshow('prediction', img_out)
              cv2.imwrite(DIR_IMGS_OUT+ '/{:04}'.format(idx_img) + '.jpg', img_out)
              
              if cv2.waitKey(1) & 0xFF == ord('\x1b'):
                  break
          else:
              break
           
          idx_img += 1

if MODE == 2:
    filenames = sorted(os.listdir(DIR_TEST))
    with tf.device(DEVICE):
        for filename in filenames:
            # read and process image
            path = os.path.join(DIR_TEST, filename)
            img = cv2.imread(path)
            img = process_img(img)
            
            # predict
            output = model.predict(img)
            
            # write images
            img_out = draw_output(img[0], output[0].numpy())
            path_out = os.path.join(DIR_IMGS_OUT, filename)
            cv2.imwrite(path_out, cv2.cvtColor(img_out, cv2.COLOR_RGB2BGR))
            print('Saved image')
