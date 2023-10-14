#import necessary libraries
import numpy as np
import cv2

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

def part5a():
    return

def part5b():
    return

imagepath= "test-field.jpg"
image = cv2.imread(imagepath)
bw_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
width = len(image)
height = len(image[0])

print("Hello welcome to COMP-6341 Assignment 1 by Iymen Abdella, Student ID: 40218280. February 1st 2022!", end="\n")

#Part5A
print("\n --------------------- Part 5 A ------------------------- \n", end="\n")
cv2.imshow("Part 5A: Non Maximum Suppression", part5a())

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

cv2.imshow("Part 5B : Canny Edge Detection", part5b(lowerThresh,upperThresh))
cv2.waitKey()

cv2.destroyAllWindows()
