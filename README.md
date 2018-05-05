# YOLOv2 : Object Detection

This repository is an implementation of YOLOv2. YOLOv2 is a state of the art Object Detection Neural Network.

There are two different versions of this Network:
1. Trained on [Pascal VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)
2. Trained on [Multiview 3D Hand Pose Dataset](http://www.rovit.ua.es/dataset/mhpdataset/)

The weights for these can be downloaded from [Model checkpoint for Object Detection VOC](http://bit.ly/2KC9pdH) and [Model checkpoint for Hand Detection](http://bit.ly/2wdYIL1).

## Requirements
1. Python 3.5+
2. Tensorflow version 1.7.0 and its requirements.
3. NumPy 1.9.0 The pip should work
4. OpenCV 3.4.0 The opencv-python pip should work
5. CUDA version 9 (Strongly Recommended)
6. cuDNN version 7.0 (Recommended)

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
This will currently work for the dummy data provided.

## Results
The mAP achieved with this implementation of YOLO is **0.82**.

Here are some visual results obtained by the network.

![Object detection on VOC](https://github.com/pmkalshetti/object_detection/blob/master/data/imgs_out/000022.jpg?raw=true)
![Hand detection](https://github.com/pmkalshetti/object_detection/blob/master/hand_detection/imgs_out/img_out.jpg?raw=true)

## References
 - [YOLO9000: Better, Faster, Stronger](https://arxiv.org/abs/1612.08242)
 - [Darknet](https://pjreddie.com/darknet/yolo/)
 - https://github.com/allanzelener/YAD2K
 - [Pascal VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)
 - [Multiview 3D Hand Pose Dataset](http://www.rovit.ua.es/dataset/mhpdataset/)
