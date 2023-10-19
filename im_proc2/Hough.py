import math
from multiprocessing.sharedctypes import Value
from re import I
from cv2 import line
import numpy as np
import cv2

def part1a_HoughTransform(edge_map, line_threshold=80):
    
    # initialize the hough space using boundaries 
    # theta is between [-90, 90] because range of arctan
    # largest d value is just the hypotenuse of the entire width and height of image

    max = math.hypot(edge_map.shape[0], edge_map.shape[1])
    hough = np.zeros([2 * int(max), 181])

    #iterate through edge map and cast votes if an edge is detected
    for i, value in np.ndenumerate(edge_map): 
        if value == 255:
            for theta in range(180):
                x = i[0]
                y = i[1]

                d = int(x * np.cos(theta-90) + y*np.sin(theta-90) + max)
                hough[d][theta] +=1

    # find all top values of hough space based on threshold and save them.
    maximums = []
    for index, value in np.ndenumerate(hough):
        if value >= line_threshold:
            maximums.append([index[0] - max, index[1]-90])
    return hough, maximums

def part1c_DrawDetectedLines(lines):

    result = image

    #convert polar coordinates to cartesian and then draw
    for rho, theta in lines:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        cv2.line(result,(x1,y1),(x2,y2),(0,0,255),1)

    return result 

#global variables
imagepath= "hough/hough2.png"
image = cv2.imread(imagepath)
width, height, channels = image.shape
bw_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#variables part 1a
lowerThresh = 100
upperThresh = 180
edge_map = cv2.Canny(image, lowerThresh,upperThresh)

print("Hello welcome to COMP-6341 Assignment 2 by Iymen Abdella, Student ID: 40218280. March 1st 2022!", end="\n")

print("\n --------------------- Part 1 A ------------------------- \n", end="\n")
hough, maximums = part1a_HoughTransform(edge_map)

print("\n --------------------- Part 1 B ------------------------- \n", end="\n")
cv2.imshow("Part 1B: Hough transform", hough)
cv2.waitKey(0)

print("\n --------------------- Part 1 C ------------------------- \n", end="\n")
cv2.imshow("Part 1C: Detect lines on hough transform", part1c_DrawDetectedLines(maximums))
cv2.waitKey(0)
