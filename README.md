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


## Network Architecture
|    | Layer    | Filters | Size      | Input Layer | Input Dimension  | Output Dimension |
| -- | -------- | ------- | --------- | ----------- | ---------------- | ---------------- |
| 1  | Conv 2D  | 32      | 3 x 3 / 1 | Input       | 416 x 416 x 3    | 416 x 416 x 32   | 
| 1  | Max Pool |         | 2 x 2 / 2 | 1 Conv 2D   | 416 x 416 x 32   | 208 x 208 x 32   |
| 2  | Conv 2D  | 64      | 3 x 3 / 1 | 1 Max Pool  | 208 x 208 x 32   | 208 x 208 x 64   | 
| 2  | Max Pool |         | 2 x 2 / 2 | 2 Conv 2D   | 208 x 208 x 64   | 104 x 104 x 64   |
| 3  | Conv 2D  | 128     | 3 x 3 / 1 | 2 Max Pool  | 104 x 104 x 64   | 104 x 104 x 128  |
| 4  | Conv 2D  | 64      | 3 x 3 / 1 | 3 Conv 2D   | 104 x 104 x 128  | 104 x 104 x 64   |
| 5  | Conv 2D  | 128     | 3 x 3 / 1 | 4 Conv 2D   | 104 x 104 x 64   | 104 x 104 x 128  | 
| 5  | Max Pool |         | 2 x 2 / 2 | 5 Conv 2D   | 104 x 104 x 128  | 52  x 52  x 128  |
| 6  | Conv 2D  | 256     | 3 x 3 / 1 | 5 Max Pool  | 52  x 52  x 128  | 52  x 52  x 256  |
| 7  | Conv 2D  | 128     | 3 x 3 / 1 | 6 Conv 2D   | 52  x 52  x 256  | 52  x 52  x 128  |
| 8  | Conv 2D  | 256     | 3 x 3 / 1 | 7 Conv 2D   | 52  x 52  x 128  | 52  x 52  x 256  | 
| 8  | Max Pool |         | 2 x 2 / 2 | 8 Conv 2D   | 52  x 52  x 256  | 26  x 26  x 256  |
| 9  | Conv 2D  | 512     | 3 x 3 / 1 | 8 Max Pool  | 26  x 26  x 256  | 26  x 26  x 512  |
| 10 | Conv 2D  | 256     | 3 x 3 / 1 | 9 Conv 2D   | 26  x 26  x 512  | 26  x 26  x 256  |
| 11 | Conv 2D  | 512     | 3 x 3 / 1 | 10 Conv 2D  | 26  x 26  x 256  | 26  x 26  x 512  |
| 12 | Conv 2D  | 256     | 3 x 3 / 1 | 11 Conv 2D  | 26  x 26  x 512  | 26  x 26  x 256  |
| 13 | Conv 2D  | 512     | 3 x 3 / 1 | 12 Conv 2D  | 26  x 26  x 256  | 26  x 26  x 512  | 
| 13 | Max Pool |         | 2 x 2 / 2 | 13 Conv 2D  | 26  x 26  x 512  | 13  x 13  x 512  |
| 14 | Conv 2D  | 1024    | 3 x 3 / 1 | 13 Max Pool | 13  x 13  x 512  | 13  x 13  x 1024 |
| 15 | Conv 2D  | 512     | 3 x 3 / 1 | 14 Conv 2D  | 13  x 13  x 1024 | 13  x 13  x 512  |
| 16 | Conv 2D  | 1024    | 3 x 3 / 1 | 15 Conv 2D  | 13  x 13  x 512  | 13  x 13  x 1024 |
| 17 | Conv 2D  | 512     | 3 x 3 / 1 | 16 Conv 2D  | 13  x 13  x 1024 | 13  x 13  x 512  |
| 18 | Conv 2D  | 1024    | 3 x 3 / 1 | 17 Conv 2D  | 13  x 13  x 512  | 13  x 13  x 1024 |
| 19 | Conv 2D  | 1024    | 3 x 3 / 1 | 18 Conv 2D  | 13  x 13  x 1024 | 13  x 13  x 1024 |
| 20 | Conv 2D  | 1024    | 3 x 3 / 1 | 19 Conv 2D  | 13  x 13  x 1024 | 13  x 13  x 1024 |
| 21 | Conv 2D  | 64      | 3 x 3 / 1 | 13 Conv 2D  | 26  x 26  x 512  | 26  x 26  x 64   |
| 22 | Conv 2D  | 1024    | 3 x 3 / 1 | 20 + 21 | 13  x 13  x (1024 + 4 * 64 ) | 13  x 13  x 1024 |
| 23 | Conv 2D  | 125 = 5 x (1+4+20) | 3 x 3 / 1 | 22 Conv 2D  | 13  x 13  x 1024 | 13  x 13  x 125 |


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
