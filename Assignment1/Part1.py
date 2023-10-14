#import necessary libraries
from turtle import down
import numpy as np
import cv2

#Part1A downsampling an image without using OpenCV
def part1a(downsize_factor=1):

    #copy every nth row from original image
    downsample = image[::downsize_factor]

    #remove every column except the nth ones
    to_remove = list(range(0,height))
    del to_remove[::downsize_factor]

    return np.delete(downsample, to_remove, axis=1)

#global variables
imagepath= "santorini.jpeg"
image = cv2.imread(imagepath)
width = len(image)
height = len(image[0])

print("Hello welcome to COMP-6341 Assignment 1 by Iymen Abdella, Student ID: 40218280. February 1st 2022!", end="\n")

print("\n --------------------- Part 1 A ------------------------- \n", end="\n")

factor = int(input("Please provide a positive number as a downsizing factor and press **Enter**. \n"))
while factor <= 0:
    factor = int(input("invalid downsizing factor!! Please use a positive number (Ex: 3, 5, 7) and press **Enter** \n"))

#Part1B 
print("\n --------------------- Part 1 B ------------------------- \n", end="\n")

cv2.imshow(f"Part 1B: Displaying downsampled image by factor of {factor}", part1a(factor))
cv2.waitKey(0)

#Part1C
print("\n --------------------- Part 1 C ------------------------- \n", end="\n")
img = part1a(16) # downsampled image I16
width = len(img)
height = len(img[0])

# upsize the images to 10 times the width and height of I16:
cv2.imshow("Part 1C: interpolation techniques: nearest neighbour", 
            cv2.resize(img, dsize=(width*10, height*10), interpolation=cv2.INTER_NEAREST),)
cv2.waitKey(0)

cv2.imshow("Part 1C: interpolation techniques: bilinear interpolation",
            cv2.resize(img, dsize=(width*10, height*10), interpolation=cv2.INTER_LINEAR),)
cv2.waitKey(0)

cv2.imshow("Part 1C: interpolation techniques: bilinear Cubic", 
            cv2.resize(img, dsize=(width*10, height*10), interpolation=cv2.INTER_CUBIC),)
cv2.waitKey(0)

#close all remaining opened windows
cv2.destroyAllWindows()