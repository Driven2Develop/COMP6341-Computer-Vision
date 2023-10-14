A repo containing all the assignment work from the Computer Vision graduate course offered at Conconrdia University. 

There are two assignments, and one final project each one dealing with various computer vision and/or image processing techniques. 

## Assignment 1: 
Using python along with OpenCV and Pytorch we were required to practice various image processing techniques including:
* resizing, downsampling, and upsampling
* image translations, and filtering
* Sobel operators to find derivatives of image and ensuing orientation
* Non-max suppression as well as Canny edge detection
More details can be found within the assginment description (assignment1.pdf)

## Assignment 2
Similar to assignment 1, Python, Pytorch and OpenCV were employed to practice various image processing techniques such as:
* Hough transform for detecting the presence of shapes or lines
* Harris corner detector
* SIFT descriptors for image feature detection and matching (contrast invariant)
* Adaptive non-max suppression algorithm as an improvement to the previous assignments implementation
More details can be found within the assignment description (assignment2.pdf)

## Project
The final project challenged the students to employ more modern image processing techniques with the use of deep neural networks to achieve better results. As a prereq to execute the project, the Standford Dogs dataset must be downloaded and stored under the 'images' directory: https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset. (far too large to upload to github unforunately). A summary of the project is as follows:
* Train a deep neural net classifier using the stanford dogs dataset as input.
* The network is in a feed forward style with:
    * **Activation Functions:** Sigmoid ReLU & Identity
    * **Loss Function:** Softmax & Cross Entropy
* The real 'magic' of the neural network is the constructiuon of a deep **convolutional neural network** using Pytorch with optimizations in learning rate.
* The convolutional layer performs well with image classification problems because of its shift invariance, and its unique architecture that transforms images into feature maps. Which are better suited for analysis of local patterns within an image. Among other benefits.
More details can be found within the project-desc.pdf 
