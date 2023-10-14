#import necessary libraries
from cmath import e
import numpy as np
import cv2

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
                    [[0,0,0], [1,1,1], [1,1,1]], 
                    [[0,0,0], [0,0,0], [1,1,1]], 
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

#global variables
imagepath= "santorini.jpeg"
image = cv2.imread(imagepath)
width = len(image)
height = len(image[0])

print("Hello welcome to COMP-6341 Assignment 1 by Iymen Abdella, Student ID: 40218280. February 1st 2022!", end="\n")

#Part2A 
print("\n --------------------- Part 2 A ------------------------- \n", end="\n")
cv2.imshow("part 2A translating diagonally to top right.", part2a())  
cv2.waitKey(0)

#Part2B 
print("\n --------------------- Part 2 B ------------------------- \n", end="\n")

# get guassian filter size from user
Nsize = int(input("Please provide a positive odd number as a neighborhood size for a Gaussian filter (Ex: 3, 5, 7) and press **Enter** \n"))
while Nsize <= 0 or Nsize%2 != 1 :
    Nsize = int(input("invalid neighbourhood size!! Please use a positive odd number (Ex: 3, 5, 7) and press **Enter** \n"))

# get scale factor from user
scale = int(input("Please provide a positive number as a scale for a Gaussian filter (Ex: 2, 4, 8) and press **Enter** \n"))
while scale <= 0 :
    scale = int(input("invalid scale!! Please use a positive number (Ex: 2, 4, 6) and press **Enter** \n"))

cv2.imshow(f"part 2B: Applied Gaussian filter with filter size {Nsize} and scale {scale}", part2b(Nsize, scale))
cv2.waitKey(0)

#Part2C
print("\n --------------------- Part 2 C ------------------------- \n", end="\n")

# get guassian filter size from user
Nsize = int(input("Please provide a positive odd number gretaer than 1 as a neighborhood size for a Gaussian filter (Ex: 3, 5, 7) and press **Enter**. Default is 3\n"))
while Nsize <= 0 or Nsize%2 != 1 :
    Nsize = int(input("invalid neighbourhood size!! Please use a positive odd number (Ex: 3, 5, 7) and press **Enter** \n"))

# get alpha for first gaussian filter from user
alpha = int(input("Please provide a positive number as a scale for first Gaussian filter (Ex: 2, 4, 8) and press **Enter** \n"))
while alpha <= 0 :
    alpha = int(input("invalid scale !! Please use a positive number (Ex: 2, 4, 6) and press **Enter** \n"))

# get beta for second gaussian filter from user user
beta = int(input("Please provide a positive number as a scale for second Gaussian filter (Ex: 2, 4, 8) and press **Enter** \n"))
while beta <= 0 :
    beta = int(input("invalid scale!! Please use a positive number (Ex: 2, 4, 6) and press **Enter** \n"))

cv2.imshow(f"part 2C: Applied Difference of Gaussian filter with filter size {Nsize}, first scale {alpha}, and second scale {beta}.", part2c(Nsize, alpha, beta))
cv2.waitKey(0)
cv2.destroyAllWindows()