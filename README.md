# TIHAN-MILAN
## 1 Problem Statement
Detecting and classifying traffic lights for autonomous driving and traffic management
is a crucial component of modern intelligent transportation systems.
This task involves using computer vision and machine learning techniques to
identify and categorize traffic lights in a given environment.
### 1.1 Objective
The primary goal of detecting and classifying traffic lights is to enable autonomous
vehicles to understand and respond to traffic signals, ensuring safe
and efficient navigation on the road.
### 1.2 Problem Statement
Detecting and classifying traffic lights involves two main aspects:
  1. Traffic Light Detection: Locating the position of traffic lights in the
camera’s field of view.
  2. Traffic Light Classification: Identifying the state of each traffic light,
i.e., whether it’s red, yellow, green.

## 2 Detection
We use [MobileNetv1](https://arxiv.org/pdf/1704.04861.pdf) model.

MobileNet SSD v1 is primarily used for the task of object detection, which involves identifying and locating objects of interest within an image or video frame. 

It can detect multiple object classes simultaneously. 

In the COCO dataset, the 10th class is traffic lights

## 3 Color Classification
We use a simple algorithm to find the color of the traffic light
  1. Convert image to HSV.
  2. Use only the pixels which have high saturation.
  3. Use the hue values to map to a color.
