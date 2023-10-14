#import necessary libraries
import random
import numpy as np
import matplotlib
import cv2
import torch
import torchvision
import torchvision.datasets as ds

rootDir = './downloaded-datasets' 

print("Hello welcome to COMP-6341 Assignment 1 by Iymen Abdella, Student ID: 40218280. February 1st 2022!", end="\n")

#Part4A
print("\n --------------------- Part 4 A ------------------------- \n", end="\n")
#download the training dataset
trainset = ds.CIFAR10(root=rootDir, download=True, transform=None, train=True)

#download the testing dataset
testset = ds.CIFAR10(root=rootDir, download=True, transform=None, train=False)

#Part4B 
print("\n --------------------- Part 4 B ------------------------- \n", end="\n")


for i in range(len(trainset)):
    cv2.imshow(trainset[i][random.randint(0, len(trainset[0]))])
    cv2.waitKey(0)

cv2.destroyAllWindows()