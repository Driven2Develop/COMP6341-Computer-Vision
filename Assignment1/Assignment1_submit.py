#import necessary libraries
from cv2 import threshold
import numpy as np
import cv2
import matplotlib
# import torch
# import torchvision
# import torchvision.datasets as ds

#Part1A downsampling an image without using OpenCV
def part1a(downsize_factor=1):

    #copy every nth row from original image
    downsample = image[::downsize_factor]

    #remove every column except the nth ones
    to_remove = list(range(0,height))
    del to_remove[::downsize_factor]

    return np.delete(downsample, to_remove, axis=1)

#function for creating a gaussian kernel for part 2B and 2C
# source: https://numpy.org/doc/stable/reference/random/generated/numpy.random.multivariate_normal.html
def createGaussKernel(size=3, sigma=1):
    centre = [int((size-1)/2), int((size-1)/2)] #since gauss kernel is an square, centre is just the middle.
    gauss = np.zeros([size,size,3])

    for i in range(size):
        for j in range(size):
            gauss[i][j] = [ np.e**((-1)*(np.square(i-centre[0]) + np.square(j-centre[1])) / (2*np.square(sigma))) / (sigma*np.sqrt(2*np.pi)),
                            np.e**((-1)*(np.square(i-centre[0]) + np.square(j-centre[1])) / (2*np.square(sigma))) / (sigma*np.sqrt(2*np.pi)) ,
                            np.e**((-1)*(np.square(i-centre[0]) + np.square(j-centre[1])) / (2*np.square(sigma))) / (sigma*np.sqrt(2*np.pi)) ]

    return gauss

def createGaussKernel_bw(size = 3, sigma = 1):
    centre = [int((size-1)/2), int((size-1)/2)] #since gauss kernel is an square, centre is just the middle.
    gauss = np.zeros([size,size])

    for i in range(size):
        for j in range(size):
            gauss[i][j] = np.e**((-1)*(np.square(i-centre[0]) + np.square(j-centre[1])) / (2*np.square(sigma))) / (sigma*np.sqrt(2*np.pi))

    return gauss
# returns a neighbourhood of length size, centerred around index (row, col) 
# pixels outside border of image are set to 0
# only odd number sizes are accepted because they have a center -- 1, 2, 3, 5, 
# source: https://stackoverflow.com/questions/22550302/find-neighbors-in-a-matrix/22550933
def neighborhood(size, row, col):
    radius = int((size-1)/2)

    return [[image[i][j] if  i >= 0 and i < width and j >= 0 and j < height else [0,0,0]
                for j in range(col-1-radius, col+radius)]
                    for i in range(row-1-radius, row+radius)]

# simple helper method for multiplying two rgb vectors:
# a1 * b1
# a2 * b2
# a3 * b3
def multRgb(vec1, vec2):
    if(len(vec1) == len(vec2)):
        return [vec1[i]*vec2[i] for i in range(len(vec1))]
    else:
        raise Exception("vectors are not same length")

# simple helper method for adding two rgb vectors:
# a1 + b1
# a2 + b2
# a3 + b3
def addRgb(vec1, vec2):

        return [vec1[i]+vec2[i] for i in range(len(vec1))]

# how to apply kernel for NxN neighbourhood
def applyKernel(kernel, target):
    if (len(kernel) != len(target)) and (len(kernel[0]) != len(target[0])) :
        raise Exception("kernel matrix and target matrix are not the same size")

    res = [[multRgb(kernel[i][j], target[i][j]) for i in range(len(kernel))] for j in range(len(kernel[0]))]
    sum = [0,0,0]
    for i in range(len(res)):
        for j in range(len(res[0])):
            sum = addRgb(res[i][j], sum)

    #floor and ceiling of r,g,b values before returning sum
    for k in range(len(sum)):
        if sum[k] > 255:
            sum[k] = 255
        else:
            if sum[k] < 0:
                sum[k] = 0
        #round to an integer before returning
        sum[k] = int(round(sum[k]))
    return sum

def part2a():
    #shifting kernel
    # 0 1 1
    # 0 0 1
    # 0 0 0 
    # working in rgb
    shift_kernel = [ 
                    [[0,0,0], [0,0,0], [1,1,1]], 
                    [[0,0,0], [0,0,0], [0,0,0]], 
                    [[0,0,0], [0,0,0], [0,0,0]]
                ]
    res = np.zeros([width, height, 3], np.uint8)

    #apply kernel to the image and save as result
    for i in range(width):
        for j in range(height):
            res[i][j] = applyKernel(shift_kernel, neighborhood(3,i,j))

    return res

#TODO: test with other sizes and scales
def part2b(Nsize=3, scale=1):
    gauss = np.zeros([width, height, 3], np.uint8)
    
    #calculate gaussian filter
    gauss_kernel = createGaussKernel(Nsize, sigma=scale)
    
    #apply kernel to the image
    for i in range(width):
        for j in range(height):
            gauss[i][j] = applyKernel(gauss_kernel, neighborhood(Nsize,i,j))
    
    return gauss

def part2c(Nsize=3, alpha=1, beta=2):
    gauss = np.zeros([width, height, 3])
    diffGauss_kernel = np.zeros([Nsize, Nsize, 3])
    
    #calculate alpha and beta gaussian kernels 
    alphaGauss_kernel = createGaussKernel(Nsize, sigma=beta)
    betaGauss_kernel = createGaussKernel(Nsize, sigma=alpha)

    #get the new kernel as the difference between alpha and beta
    for i in range(Nsize):
        for j in range(Nsize):
            diffGauss_kernel[i][j] = [abs(alphaGauss_kernel[i][j][0] - betaGauss_kernel[i][j][0]),
                                      abs(alphaGauss_kernel[i][j][1] - betaGauss_kernel[i][j][1]),
                                      abs(alphaGauss_kernel[i][j][2] - betaGauss_kernel[i][j][2])
                                    ]       

    #apply kernel to the image
    for i in range(width):
        for j in range(height):
            gauss[i][j] = applyKernel(diffGauss_kernel, neighborhood(Nsize,i,j))

    return gauss

# how to apply kernel for NxN neighbourhood
def applyKernel_bw(kernel, target):
    if (len(kernel) != len(target)) or (len(kernel[0]) != len(target[0])) :
        raise Exception("kernel matrix and target matrix are not the same size")

    res = [[kernel[i][j] * target[i][j] for i in range(len(kernel))] for j in range(len(kernel[0]))]
    sum = 0
    for i in range(len(res)):
        for j in range(len(res[0])):
            sum += res[i][j]

    #floor and ceiling of values before returning sum
    if sum > 255:
        sum = 255
    else:
        if sum < 0:
            sum = 0

    #round to an integer before returning
    return int(round(sum))

def neighborhood_bw(size, row, col):
    radius = int((size-1)/2)

    return [[bw_image[i][j] if  i >= 0 and i < width and j >= 0 and j < height else 0
                for j in range(col-1-radius, col+radius)]
                    for i in range(row-1-radius, row+radius)]

def part3a_horizontal():
    sobel = np.zeros([width, height], np.uint8)
    
    #Sobel filter for x axis
    # 1 0 -1
    # 2 0 -2
    # 1 0 -1
    sobel_x_kernel =[[1, 0, -1], 
                     [2, 0, -2], 
                     [1, 0, -1]
                    ]
    
    #apply kernel to the image
    for i in range(width):
        for j in range(height):
            sobel[i][j] = applyKernel_bw(sobel_x_kernel, neighborhood_bw(3, i, j))
    
    return sobel

def part3a_vertical():
    sobel = np.zeros([width, height], np.uint8)
    
    #Sobel filter for y axis
    # 1 2 1
    # 0 0 0
    # -1 -2 -1
    sobel_y_kernel =[[1, 2, 1], 
                     [0, 0, 0], 
                     [-1, -2, -1]
                    ]
    
    #apply kernel to the image
    for i in range(width):
        for j in range(height):
            sobel[i][j] = applyKernel_bw(sobel_y_kernel, neighborhood_bw(3,i,j))
    
    return sobel

# get the orientation -- actan(Gy/Gx)
def part3b():
    sobel_x = part3a_horizontal()
    sobel_y = part3a_vertical()

    res = np.zeros([len(sobel_x), len(sobel_x[0])])

    for i in range(len(sobel_x)):
        for j in range (len(sobel_x[0])):
            ori = np.round(np.arctan2(sobel_y[i][j], sobel_x[i][j]))
            if ori > 255:
                res[i][j] = 255
            else:
                if ori < 0:
                    res[i][j] = 0
                else:
                    res[i][j] = ori
    return res

#calculate the magnitude using -- (Gy^2 + Gx^2)^1/2
def part3c():
    sobel_x = part3a_horizontal()
    sobel_y = part3a_vertical()

    res = np.zeros([len(sobel_x), len(sobel_x[0])])

    for i in range(len(sobel_x)):
        for j in range (len(sobel_x[0])):
            mag = np.round(np.sqrt(sobel_x[i][j]**2 + sobel_y[i][j]**2))
            if mag > 255:
                res[i][j] = 255
            else:
                if mag < 0:
                    res[i][j] = 0
                else:
                    res[i][j] = mag

    return res

#if no orientation map is passed, create one using 3b
def part5a(orient):

    if orient is None: orient = part3b()

    #first obtain the orientation map image using part 3b, pad image all around with 0's to avoid errors in algo
    padded = np.pad(bw_image, 1, mode='constant')
    res = np.zeros([width,height])

    for i in range(width):
        for j in range(height):
            r = orient[i][j]%(2*np.pi)
            if r >= (15 * np.pi / 8) or r < (np.pi / 8) or (7*np.pi/8) <= r < (9*np.pi/8): # horizontal comparison 
                res[i][j] = max(padded[i][j], padded[i-1][j], padded[i+1][j])

            if (np.pi/8) <= r < (3*np.pi/8) or (9*np.pi/8) <= r < (11*np.pi/8): #Top right to bottom left comparison
                res[i][j] = max(padded[i][j], padded[i+1][j-1], padded[i-1][j+1]) 
                
            if (3*np.pi/8) <= r < (5*np.pi/8) or (11*np.pi/8) <= r < (13*np.pi/8): #vertical comparison
                res[i][j] = max(padded[i][j], padded[i][j-1], padded[i][j+1])

            if (5*np.pi/8) <= r < (7*np.pi/8) or (13*np.pi/8) <= r < (15*np.pi/8): #Top left to bottom right comparison
                res[i][j] = max(padded[i][j], padded[i-1][j-1], padded[i+1][j+1]) 
            
    return res

def part5b(lowerThresh, upperThresh): #upper hysteresis and lower hysteresis vlaues

    thresh_m = 25

    #smooth image using gaussian kernel in bw
    canny_image = np.zeros([width, height])
    gauss = createGaussKernel_bw(size=3, sigma=2)
    
    for i in range(width):
        for j in range(height):
            canny_image[i][j] = applyKernel_bw(gauss, neighborhood_bw(3, i, j))

    #calculate the magnitude of the gradient from part 3c
    mag = part3c()

    #calculate the orientation of thresholded pixels larger than threshold value
    sobel_x = part3a_horizontal()
    sobel_y = part3a_vertical()

    ori_map = np.zeros([width, height])

    for i in range(len(mag)):
        for j in range(len(mag[0])):
            if (mag[i][j] > thresh_m):
                ori = np.round(np.arctan2(sobel_y[i][j], sobel_x[i][j]))
                if ori > 255:
                    ori_map[i][j] = 255
                else:
                    if ori < 0:
                        ori_map[i][j] = 0
                    else:
                        ori_map[i][j] = ori
            else:
                ori_map[i][j] = 0    
    
    #thin the image using non-maximum suppression
    ori_map = part5a(ori_map)
    res = np.zeros([len(canny_image), len(canny_image[0])])
    
    #perform the hysteresis
    for i in range(len(canny_image)):
        for j in range(len(canny_image[0])):

            if mag[i][j] > upperThresh: # keep pixels higher than ceiling
                res[i][j] = canny_image[i][j]
    
    #second pass of hysteresis
    for i in range(len(canny_image)):
        for j in range(len(canny_image[0])):

            radius = int((3-1)/2) 

            #create neighbourhood 3x3
            hood = [[res[a][b] if  a >= 0 and a < width and b >= 0 and b < height else 0
                        for a in range(j-1-radius, j+radius)]
                            for b in range(i-1-radius, i+radius)]

            if mag[i][j] > lowerThresh and 1 in hood: #if magnitude is higher than the floor check to see if it has a neighbor that passed first round
                res[i][j] = canny_image[i][j]

    return res

#global variables
imagepath= "test-field.jpg"
image = cv2.imread(imagepath)
width = len(image)
height = len(image[0])
bw_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
rootDir = './downloaded-datasets' 

# print("Hello welcome to COMP-6341 Assignment 1 by Iymen Abdella, Student ID: 40218280. February 1st 2022!", end="\n")

# print("\n --------------------- Part 1 A ------------------------- \n", end="\n")

# factor = int(input("Please provide a positive number as a downsizing factor and press **Enter**. \n"))
# while factor <= 0:
#     factor = int(input("invalid downsizing factor!! Please use a positive number (Ex: 3, 5, 7) and press **Enter** \n"))

# #Part1B 
# print("\n --------------------- Part 1 B ------------------------- \n", end="\n")

# cv2.imshow(f"Part 1B: Displaying downsampled image by factor of {factor}", part1a(factor))
# cv2.waitKey(0)

# #Part1C
# print("\n --------------------- Part 1 C ------------------------- \n", end="\n")
# img = part1a(16) # downsampled image I16

# # upsize the images to 10 times the width and height of I16:
# cv2.imshow("Part 1C: interpolation techniques: nearest neighbour", 
#             cv2.resize(img, dsize=(len(img)*10, len(img[0])*10), interpolation=cv2.INTER_NEAREST),)
# cv2.waitKey(0)

# cv2.imshow("Part 1C: interpolation techniques: bilinear interpolation",
#             cv2.resize(img, dsize=(len(img)*10, len(img[0])*10), interpolation=cv2.INTER_LINEAR),)
# cv2.waitKey(0)

# cv2.imshow("Part 1C: interpolation techniques: bilinear Cubic", 
#             cv2.resize(img, dsize=(len(img)*10, len(img[0])*10), interpolation=cv2.INTER_CUBIC),)
# cv2.waitKey(0)

# #Part2A 
# print("\n --------------------- Part 2 A ------------------------- \n", end="\n")
# cv2.imshow("part 2A translating diagonally to top right.", part2a())  
# cv2.waitKey(0)

# #Part2B 
# print("\n --------------------- Part 2 B ------------------------- \n", end="\n")

# # get guassian filter size from user
# Nsize = int(input("Please provide a positive odd number as a neighborhood size for a Gaussian filter (Ex: 3, 5, 7) and press **Enter** \n"))
# while Nsize <= 0 or Nsize%2 != 1 :
#     Nsize = int(input("invalid neighbourhood size!! Please use a positive odd number (Ex: 3, 5, 7) and press **Enter** \n"))

# # get scale factor from user
# scale = int(input("Please provide a positive number as a scale for a Gaussian filter (Ex: 2, 4, 8) and press **Enter** \n"))
# while scale <= 0 :
#     scale = int(input("invalid scale!! Please use a positive number (Ex: 2, 4, 6) and press **Enter** \n"))

# cv2.imshow(f"part 2B: Applied Gaussian filter with filter size {Nsize} and scale {scale}", part2b(Nsize, scale))
# cv2.waitKey(0)

# #Part2C
# print("\n --------------------- Part 2 C ------------------------- \n", end="\n")

# # get guassian filter size from user
# Nsize = int(input("Please provide a positive odd number gretaer than 1 as a neighborhood size for a Gaussian filter (Ex: 3, 5, 7) and press **Enter**. Default is 3\n"))
# while Nsize <= 0 or Nsize%2 != 1 :
#     Nsize = int(input("invalid neighbourhood size!! Please use a positive odd number (Ex: 3, 5, 7) and press **Enter** \n"))

# # get alpha for first gaussian filter from user
# alpha = int(input("Please provide a positive number as a scale for first Gaussian filter (Ex: 2, 4, 8) and press **Enter** \n"))
# while alpha <= 0 :
#     alpha = int(input("invalid scale !! Please use a positive number (Ex: 2, 4, 6) and press **Enter** \n"))

# # get beta for second gaussian filter from user user
# beta = float(input("Please provide a positive number as a scale for second Gaussian filter (Ex: 2, 4, 8) and press **Enter** \n"))
# while beta <= 0 :
#     beta = float(input("invalid scale!! Please use a positive number (Ex: 2, 4, 6) and press **Enter** \n"))

# cv2.imshow(f"part 2C: Applied Difference of Gaussian filter with filter size {Nsize}, first scale {alpha}, and second scale {beta}.", part2c(Nsize, alpha, beta))
# cv2.waitKey(0)

#Part3A 
print("\n --------------------- Part 3 A ------------------------- \n", end="\n")
cv2.imshow("Part 3A: Sobel Operator w.r.t. X", part3a_horizontal())
cv2.waitKey(0)

cv2.imshow("Part 3A: Sobel Operator w.r.t. Y", part3a_vertical())
cv2.waitKey(0)

#Part3B 
print("\n --------------------- Part 3 B ------------------------- \n", end="\n")
cv2.imshow("Part 3B: Orientation Map", part3b())
cv2.waitKey()

#Part3C 
print("\n --------------------- Part 3 C ------------------------- \n", end="\n")
cv2.imshow("Part 3C: Gradient Magnitude", part3c())
cv2.waitKey()

#Part3D 
print("\n --------------------- Part 3 D ------------------------- \n", end="\n")
# get optional lower threshold from user
lowerThresh = 100;
if lowerThresh == 100:
    lowerThresh = int(input("Please provide a positive number as a LOWER threshold for Canny edge detection. Default is 100 (Ex: 120, 140, 180) and press **Enter** \n"))
    while lowerThresh <= 0 :
        lowerThresh = int(input("invalid threshold size!! Default is 100 (Ex: 120, 140, 180) and press **Enter** \n"))

# get optional upper threshold from user
upperThresh = 180;
if upperThresh == 180:
    upperThresh = int(input("Please provide a positive number as an UPPER threshold for Canny edge detection. Default is 180 (Ex: 120, 140, 180) and press **Enter** \n"))
    while upperThresh <= 0 :
        upperThresh = int(input("invalid threshold!! Default is 180 (Ex: 120, 140, 180) and press **Enter** \n"))

cv2.imshow("Part 3D: Canny Edge Detection", cv2.Canny(image, lowerThresh,upperThresh))
cv2.waitKey()

#Part4A
print("\n --------------------- Part 4 A ------------------------- \n", end="\n")
#download the training dataset
#trainset = ds.CIFAR10(root=rootDir, download=True, transform=None, train=True)

#download the testing dataset
#set = ds.CIFAR10(root=rootDir, download=True, transform=None, train=False)

#Part4B 
print("\n --------------------- Part 4 B ------------------------- \n", end="\n")

for i in range(set.__len__):
    cv2.imshow(set.__getitem__(i)) 

#Part5A
print("\n --------------------- Part 5 A ------------------------- \n", end="\n")
#cv2.imshow("Part 5A: Non Maximum Suppression", part5a())

#Part5B      
print("\n --------------------- Part 5 B ------------------------- \n", end="\n")

# get optional lower threshold from user
lowerThresh = 100;
if lowerThresh == 100:
    lowerThresh = int(input("Please provide a positive number as a LOWER threshold for Canny edge detection. Default is 100 (Ex: 120, 140, 180) and press **Enter** \n"))
    while lowerThresh <= 0 :
        lowerThresh = int(input("invalid threshold size!! Default is 100 (Ex: 120, 140, 180) and press **Enter** \n"))

# get optional upper threshold from user
upperThresh = 180;
if upperThresh == 180:
    upperThresh = int(input("Please provide a positive number as an UPPER threshold for Canny edge detection. Default is 180 (Ex: 120, 140, 180) and press **Enter** \n"))
    while upperThresh <= 0 :
        upperThresh = int(input("invalid threshold!! Default is 180 (Ex: 120, 140, 180) and press **Enter** \n"))

cv2.imshow("Part 5B : Canny Edge Detection", part5b(lowerThresh, upperThresh))
cv2.waitKey()

#close all remaining opened windows
cv2.destroyAllWindows()