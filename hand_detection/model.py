import tensorflow as tf
import tensorflow.contrib.eager as tfe
from utils import *
from constants import *
import pickle


class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.optimizer = tf.train.AdamOptimizer()
        
        # add layers
        self.conv1 = tf.keras.layers.Conv2D(32, 3, padding='same', use_bias=False)
        self.norm1 = tf.keras.layers.BatchNormalization()
        self.pool1 = tf.keras.layers.MaxPool2D()

        self.conv2 = tf.keras.layers.Conv2D(64, 3, padding='same', use_bias=False)
        self.norm2 = tf.keras.layers.BatchNormalization()
        self.pool2 = tf.keras.layers.MaxPool2D()
        
        self.conv3 = tf.keras.layers.Conv2D(128, 3, padding='same', use_bias=False)
        self.norm3 = tf.keras.layers.BatchNormalization()
        
        self.conv4 = tf.keras.layers.Conv2D(64, 1, padding='same', use_bias=False)
        self.norm4 = tf.keras.layers.BatchNormalization()
        
        self.conv5 = tf.keras.layers.Conv2D(128, 3, padding='same', use_bias=False)
        self.norm5 = tf.keras.layers.BatchNormalization()
        self.pool5 = tf.keras.layers.MaxPool2D()
        
        self.conv6 = tf.keras.layers.Conv2D(256, 3, padding='same', use_bias=False)
        self.norm6 = tf.keras.layers.BatchNormalization()
        
        self.conv7 = tf.keras.layers.Conv2D(128, 1, padding='same', use_bias=False)
        self.norm7 = tf.keras.layers.BatchNormalization()
        
        self.conv8 = tf.keras.layers.Conv2D(256, 3, padding='same', use_bias=False)
        self.norm8 = tf.keras.layers.BatchNormalization()
        self.pool8 = tf.keras.layers.MaxPool2D()
        
        self.conv9 = tf.keras.layers.Conv2D(512, 3, padding='same', use_bias=False)
        self.norm9 = tf.keras.layers.BatchNormalization()
        
        self.conv10 = tf.keras.layers.Conv2D(256, 1, padding='same', use_bias=False)
        self.norm10 = tf.keras.layers.BatchNormalization()
        
        self.conv11 = tf.keras.layers.Conv2D(512, 3, padding='same', use_bias=False)
        self.norm11 = tf.keras.layers.BatchNormalization()
        
        self.conv12 = tf.keras.layers.Conv2D(256, 1, padding='same', use_bias=False)
        self.norm12 = tf.keras.layers.BatchNormalization()
        
        self.conv13 = tf.keras.layers.Conv2D(512, 3, padding='same', use_bias=False)
        self.norm13 = tf.keras.layers.BatchNormalization()  # skip after this
        self.pool13 = tf.keras.layers.MaxPool2D()
        
        self.conv14 = tf.keras.layers.Conv2D(1024, 3, padding='same', use_bias=False)
        self.norm14 = tf.keras.layers.BatchNormalization()
        
        self.conv15 = tf.keras.layers.Conv2D(512, 1, padding='same', use_bias=False)
        self.norm15 = tf.keras.layers.BatchNormalization()
        
        self.conv16 = tf.keras.layers.Conv2D(1024, 3, padding='same', use_bias=False)
        self.norm16 = tf.keras.layers.BatchNormalization()
        
        self.conv17 = tf.keras.layers.Conv2D(512, 1, padding='same', use_bias=False)
        self.norm17 = tf.keras.layers.BatchNormalization()
        
        self.conv18 = tf.keras.layers.Conv2D(1024, 3, padding='same', use_bias=False)
        self.norm18 = tf.keras.layers.BatchNormalization()
        
        self.conv19 = tf.keras.layers.Conv2D(1024, 3, padding='same', use_bias=False)
        self.norm19 = tf.keras.layers.BatchNormalization()
        
        self.conv20 = tf.keras.layers.Conv2D(1024, 3, padding='same', use_bias=False)
        self.norm20 = tf.keras.layers.BatchNormalization()
        
        self.conv21 = tf.keras.layers.Conv2D(64, 1, padding='same', use_bias=False)  # apply on skipped connection
        self.norm21 = tf.keras.layers.BatchNormalization()
        
        self.conv22 = tf.keras.layers.Conv2D(1024, 3, padding='same', use_bias=False)
        self.norm22 = tf.keras.layers.BatchNormalization()
        # Feature Extractor Ends Here!
        
        # Detector Layer!
        self.conv23 = tf.keras.layers.Conv2D(NUM_ANCHORS*(4+1+NUM_OBJECTS), 1, padding='same')
        
    def forward(self, imgs):
        # imgs.shape = [B, IMG_OUT_H, IMG_OUT_W, 3]
        
        # for now, resize and reshape imgs to vector
        imgs = tf.image.resize_images(imgs, [416, 416])
        
        x = self.conv1(imgs)
        x = self.norm1(x)
        x = tf.nn.leaky_relu(x, alpha=0.1)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.norm2(x)
        x = tf.nn.leaky_relu(x, alpha=0.1)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.norm3(x)
        x = tf.nn.leaky_relu(x, alpha=0.1)
        
        x = self.conv4(x)
        x = self.norm4(x)
        x = tf.nn.leaky_relu(x, alpha=0.1)
        
        x = self.conv5(x)
        x = self.norm5(x)
        x = tf.nn.leaky_relu(x, alpha=0.1)
        x = self.pool5(x)
        
        x = self.conv6(x)
        x = self.norm6(x)
        x = tf.nn.leaky_relu(x, alpha=0.1)
        
        x = self.conv7(x)
        x = self.norm7(x)
        x = tf.nn.leaky_relu(x, alpha=0.1)
        
        x = self.conv8(x)
        x = self.norm8(x)
        x = tf.nn.leaky_relu(x, alpha=0.1)
        x = self.pool8(x)
        
        x = self.conv9(x)
        x = self.norm9(x)
        x = tf.nn.leaky_relu(x, alpha=0.1)
        
        x = self.conv10(x)
        x = self.norm10(x)
        x = tf.nn.leaky_relu(x, alpha=0.1)
        
        x = self.conv11(x)
        x = self.norm11(x)
        x = tf.nn.leaky_relu(x, alpha=0.1)
        
        x = self.conv12(x)
        x = self.norm12(x)
        x = tf.nn.leaky_relu(x, alpha=0.1)
        
        x = self.conv13(x)
        x = self.norm13(x)
        x = tf.nn.leaky_relu(x, alpha=0.1)
        x_skip = tf.identity(x)
        x = self.pool13(x)
        
        x = self.conv14(x)
        x = self.norm14(x)
        x = tf.nn.leaky_relu(x, alpha=0.1)
        
        x = self.conv15(x)
        x = self.norm15(x)
        x = tf.nn.leaky_relu(x, alpha=0.1)
        
        x = self.conv16(x)
        x = self.norm16(x)
        x = tf.nn.leaky_relu(x, alpha=0.1)
        
        x = self.conv17(x)
        x = self.norm17(x)
        x = tf.nn.leaky_relu(x, alpha=0.1)
        
        x = self.conv18(x)
        x = self.norm18(x)
        x = tf.nn.leaky_relu(x, alpha=0.1)
        
        x = self.conv19(x)
        x = self.norm19(x)
        x = tf.nn.leaky_relu(x, alpha=0.1)
        
        x = self.conv20(x)
        x = self.norm20(x)
        x = tf.nn.leaky_relu(x, alpha=0.1)
        
        x_skip = self.conv21(x_skip)
        x_skip = self.norm21(x_skip)
        x_skip = tf.nn.leaky_relu(x_skip, alpha=0.1)
        x_skip = tf.space_to_depth(x_skip, block_size=2)  # lossless shrinkage of feature map
        
        x = tf.concat([x_skip, x], axis=-1)  # low_level features concatenated with high_level features
        
        x = self.conv22(x)
        x = self.norm22(x)
        x = tf.nn.leaky_relu(x, alpha=0.1)
        # Feature Extractor ends here!
        
        # Detector layer
        x = self.conv23(x)
        
        # reshape output
        pred = tf.reshape(x, [-1, GRID_H, GRID_W, NUM_ANCHORS, 4+1+NUM_OBJECTS])
        
        return pred       
            
        
    def predict(self, imgs):
        '''predicts bboxes and draws them on the image'''
        # imgs.shape = [B, IMG_OUT_H, IMG_OUT_W, 3]
        
        # forward pass
        predictions = self.forward(imgs)
        
        predictions = tf.concat([predictions[...,1::-1], predictions[...,3:1:-1], predictions[...,4:]], axis=-1)
        
        # post-process to get bounding boxes
        outputs = predictions2outputs(predictions)  
        # CAUTION!!!
        # TODO: use batch multi-class nms (currently works with BATCH_SIZE=1)
        # reference: https://github.com/tensorflow/models/blob/master/research/object_detection/core/post_processing.py
        
        return outputs
        
    def load_pretrained_weights(self, dir_weights):
        for idx_layer, layer in enumerate(self.layers):
            if idx_layer == 49:  # skip final layer beacause of class mismatch
                continue
            filename = dir_weights + '/' + str(idx_layer)
            with open(filename, 'rb') as file:
                weights = pickle.load(file)
            
            layer.set_weights(weights)
        
        print('Weights loaded.')
        
        print('Weights loaded.')
