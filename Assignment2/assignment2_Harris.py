#import necessary libraries
from multiprocessing.sharedctypes import Value
from re import I
from cv2 import line, threshold
import numpy as np
import cv2
import matplotlib.pyplot as plt

def sobelx(img):

    sobel = np.zeros([width, height], np.uint8)
    
    sobel_x_kernel =[[1, 0, -1], 
                     [2, 0, -2], 
                     [1, 0, -1]
                    ]
    
    #apply kernel to the image
    for i in range(width):
        for j in range(height):
            sobel[i][j] = applyKernel_bw(sobel_x_kernel, neighborhood_bw(img,3, i, j))
    
    return sobel

def sobely(img):
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
            sobel[i][j] = applyKernel_bw(sobel_y_kernel, neighborhood_bw(img,3,i,j))
    
    return sobel

# how to apply kernel for NxN neighbourhood
def applyKernel_bw(kernel, target):

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

# returns a neighbourhood of length size, centerred around index (row, col) 
# pixels outside border of image are set to 0
# only odd number sizes are accepted because they have a center -- 1, 2, 3, 5, 
# source: https://stackoverflow.com/questions/22550302/find-neighbors-in-a-matrix/22550933
def neighborhood_bw(src, size, row, col):
    radius = int((size-1)/2)

    return [[src[i][j] if  i >= 0 and i < width and j >= 0 and j < height else 0
                for j in range(col-1-radius, col+radius)]
                    for i in range(row-1-radius, row+radius)]

def part2a_HarrisCornerDetection(src, thresh=20, k=0.04):

    res = np.zeros([width, height])
    
    #compute gradients after smooothing
    bw_blur = cv2.GaussianBlur(src, (5,5), 1, 1, cv2.BORDER_DEFAULT)
    sobel_x = sobelx(bw_blur)
    sobel_y = sobely(bw_blur)

    #compute multiplication and then smooth with gaussian
    sobel_xx = cv2.GaussianBlur(sobel_x*sobel_x, (5,5), 1, 1, cv2.BORDER_DEFAULT) 
    sobel_yy = cv2.GaussianBlur(sobel_y*sobel_y, (5,5), 1, 1, cv2.BORDER_DEFAULT) 
    sobel_xy = cv2.GaussianBlur(sobel_x*sobel_y, (5,5), 1, 1, cv2.BORDER_DEFAULT) 

    #compute Harris matrix for 3x3 neighbourhood
    for index, value in np.ndenumerate(src):
        row = index[0]
        col = index[1]

        #compute sums of components using list comprehension 
        sum_xx = sum(sum(neighborhood_bw(sobel_xx,3,row,col),[]))
        sum_xy = sum(sum(neighborhood_bw(sobel_xy,3,row,col),[]))
        sum_yy = sum(sum(neighborhood_bw(sobel_yy,3,row,col),[]))

        #combine to form the Harris matrix and compute the corner response. 
        det = (sum_xx * sum_yy) - (sum_xy**2)
        trace = sum_xx + sum_yy        
        
        corner_response = det - k*(trace**2)
        res[row][col] = corner_response
    
    #normalize and threshold before returning result:
    oldMax = np.amax(res)

    for index, value in np.ndenumerate(res):
        normalized_value = value*255/oldMax
        if normalized_value > thresh:
            res[index] = normalized_value
        else:
            res[index] = 0
        
    return sobel_x, sobel_y, sobel_xy, res

def part3a_MatchDescriptors(src, dst):

    src_bw = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    dst_bw = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

    #get keypoints and descriptors using SIFT
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints_src, descriptors_1 = sift.detectAndCompute(src_bw,None)
    keypoints_dst, descriptors_2 = sift.detectAndCompute(dst_bw,None)

    matches = []
    # get the minmimum SSD between descriptor 1 and descriptor 2
    for desc_1 in descriptors_1:
        ssd = [np.sum((desc_1[:]-j[:])**2)
                    for j in descriptors_2]
        
        min = np.amin(ssd)
        min_2 = np.amin(np.array(ssd)[ssd != np.amin(ssd)])
        index_min = np.argmin(ssd)

        # match features using the Ratio test
        ratio = min/min_2
        if(ratio < 0.90):
            matches.append([desc_1,descriptors_2[index_min]])
    
    # Part 3b display key points
    src_kp = cv2.drawKeypoints(src_bw, keypoints_src, src)
    dst_kp = cv2.drawKeypoints(dst_bw, keypoints_dst, dst)
    cv2.imshow("Key points of source image", src_kp)
    cv2.imshow("Key points of destination image", dst_kp)
    cv2.waitKey(0)

    # Part 3c draw matched keypoints but only top 50
    matched_img = cv2.drawMatches(src_bw, keypoints_src, dst_bw, keypoints_dst, matches[:50], dst_bw, flags=2, matchesThickness = 1)
    cv2.imshow("matched key points", matched_img)
    
    return 

def part5a_DescribeFeatures_ContrastInvariant(): 
    return 

def part5b_AdaptiveNon_MaxSuppression(): 
    return 

#variables for part 2
imagepath= "hough/hough1.png"
image = cv2.imread(imagepath)
width, height, channels = image.shape
bw_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#variables for part 3
yosemite_img1 = cv2.imread("image_sets/yosemite/Yosemite1.jpg")
yosemite_img2 = cv2.imread("image_sets/yosemite/Yosemite2.jpg")

print("Hello welcome to COMP-6341 Assignment 2 by Iymen Abdella, Student ID: 40218280. March 1st 2022!", end="\n")

print("\n --------------------- Part 2 A ------------------------- \n", end="\n")
Ix, Iy, Ixy, res =  part2a_HarrisCornerDetection(src=bw_image)
cv2.waitKey(0)

print("\n --------------------- Part 2 B ------------------------- \n", end="\n")
cv2.imshow("part 2B: Ix", Ix)
cv2.imshow("part 2B: Iy", Iy)
cv2.imshow("part 2B: Ixy", Ixy)
cv2.waitKey(0)

print("\n --------------------- Part 2 C ------------------------- \n", end="\n")
cv2.imshow(f"part 2C: results of corner strength response", res)
cv2.waitKey(0)

print("\n --------------------- Part 2 D ------------------------- \n", end="\n")
image[res>0.04*res.max()]=[0,0,255] # draw Harris corner points in red
res = res.astype(int)
#keypoints = cv2.drawKeypoints(image, res, np.array([]), color=(0,255,0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow(f"part 2D: using openCV draw key points", image)
cv2.waitKey(0)

print("\n --------------------- Part 3 A ------------------------- \n", end="\n")
cv2.imshow("Part 3A: SIFT-like descriptors and matches ", part3a_MatchDescriptors(src = yosemite_img1, dst=yosemite_img2))
cv2.waitKey(0)

# #close all remaining opened windows
cv2.destroyAllWindows()