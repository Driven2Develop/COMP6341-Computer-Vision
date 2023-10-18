A repo containing practice assignments based around the topic of Computer Vision. 

All of them were written in python using the OpenCV or Pytorch library

## Assignment 1: 
Image processing techniques using python including:
* resizing, downsampling, and upsampling
* image translations, and filtering
* Sobel operators to find derivatives of image and ensuing orientation
* Non-max suppression as well as Canny edge detection

## Assignment 2
More image processing techniques including:
* Hough transform for detecting the presence of shapes or lines
* Harris corner detector
* SIFT descriptors for image feature detection and matching (contrast invariant)
* Adaptive non-max suppression algorithm as an improvement to the previous assignments implementation

## Project
Includes more modern image processing techniques with the use of deep neural networks to achieve better results. 

As a prereq to execute the project, the Standford Dogs dataset must be downloaded and stored under the 'images' directory: https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset. (far too large to upload to github unforunately). A summary of the project is as follows:
* Train a deep neural net classifier using the stanford dogs dataset as input.
* The network is in a feed forward style with:
    * **Activation Functions:** Sigmoid ReLU & Identity
    * **Loss Function:** Softmax & Cross Entropy
* The real 'magic' of the neural network is the constructiuon of a deep **convolutional neural network** using Pytorch with optimizations in learning rate.
* The convolutional layer performs well with image classification problems because of its shift invariance, and its unique architecture that transforms images into feature maps. Which are better suited for analysis of local patterns within an image. Among other benefits.
* The program is captured within the jupyter notebook which is the skeleton of the script. 
