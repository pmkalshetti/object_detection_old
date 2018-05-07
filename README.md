# YOLOv2 : Object Detection

YOLOv2 is a state of the art Object Detection Neural Network. This repository is an implementation of YOLOv2 from scratch in tensorflow (eager).

There are two different versions of this Network:
1. Trained on [Pascal VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)
2. Trained on [Multiview 3D Hand Pose Dataset](http://www.rovit.ua.es/dataset/mhpdataset/)

The weights for these can be downloaded from [Model checkpoint for Object Detection VOC](http://bit.ly/2KC9pdH) and [Model checkpoint for Hand Detection](http://bit.ly/2wdYIL1).


## Requirements
1. [Python](https://www.python.org/) 3.5+
2. [Tensorflow](https://www.tensorflow.org/) version 1.7.0 and its requirements.
3. [NumPy](http://www.numpy.org/) 1.9.0.
4. [OpenCV](https://opencv.org/) 3.4.0 The opencv-python pip should work
5. CUDA version 9 (Strongly Recommended)
6. cuDNN version 7.0 (Recommended)


## Datasets
The datasets used to train the network are listed below:

| Name of Dataset | No. of Images | No. of Classes |
|-----------------|---------------|----------------|
| Pascal VoC 2013 | 17125         | 20             |
| Multiview 3D Hand Pose Dataset | | 1         |


## First time setup
```bash
git clone https://github.com/pmkalshetti/object_detection.git
cd object_detection
pip install -r requirements.txt
```


## Model - Generating data, Training and Prediction
* Make necessary changes in `constants.py` to reflect your dataset (both train and prediction) then run the following commands to make TFRecords, that will be loaded into the train.py.
```bash
    cd src
    python3 write_data_to_TFRecords.py
```
* Train the model using the following command. This might take a long time.
```bash
    python3 train.py
```
* Do the prediction. If not done yet, make changes to reflect your input to do the prediction. Make changes in `constants.py` to reflect the output directory for network predictions. Then run the following to do the inference.
```
    python3 predict.py
```
This will currently work for the dummy data provided, if you do not make any changes in `constants.py`.


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
