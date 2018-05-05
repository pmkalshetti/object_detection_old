# YOLOv2 : Object Detection

This repository is an implementation of YOLOv2.

The original paper can be found out at [YOLO9000: Better, Faster, Stronger](https://arxiv.org/abs/1612.08242) and more details can be found out at the original author's webpage [Darknet](https://pjreddie.com/darknet/yolo/). YOLOv2 the current state of the art Object Detection Neural Network.

There are two different versions of this Network:
1. Trained on [Pascal VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)
2. Trained on [Multiview 3D Hand Pose Dataset](http://www.rovit.ua.es/dataset/mhpdataset/)

The weights for them can be downloaded from [Model checkpoint for Object Detection](http://bit.ly/2KC9pdH) and [Model checkpoint for Hand Detection](http://bit.ly/2wdYIL1).

## Requirements
1. Python 3.5+
2. Tensorflow version 1.7.0 and its requirements.
3. NumPy 1.9.0 The pip should work
4. OpenCV 3.4.0 The opencv-python pip should work
5. CUDA version 9 (Strongly Recommended)
6. cuDNN version 7.0 (Recommended)

## Directory Structure
|-- data<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|-- annotations<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|-- data_tfrecords<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|-- imgs_out<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|-- input<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|-- test<br>
|-- docs<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|-- presetation.pdf<br>
|-- hand_detection<br>
|-- notebooks<br>
|-- src<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|-- constants.py<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|-- model.py<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|-- predict.py<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|-- train.py<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|-- utils.py<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|-- write_data_to_TFRecords.py<br>
|-- README.md<br>

## First time setup
```bash
git clone https://github.com/pmkalshetti/object_detection.git
cd object_detection
pip install -r requirements.txt
```

## Model - Generating data, Training and Prediction
```bash
python3 write_data_to_TFRecords.py
python3 train.py
python3 predict.py
```

## Results
The mAP achieved with this implementation of YOLO is **0.82**.

## Reference
 - https://github.com/allanzelener/YAD2K
 - [YOLO9000: Better, Faster, Stronger](https://arxiv.org/abs/1612.08242)
 - [Darknet](https://pjreddie.com/darknet/yolo/)
 - [Pascal VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)
 - [Multiview 3D Hand Pose Dataset](http://www.rovit.ua.es/dataset/mhpdataset/)
