# from cv2 import threshold
import numpy as np 
import cv2
from matplotlib import pyplot as plt


# # to access the pixel values 
# img = cv2.imread('test_image.png')
# px = img[100,100]
# print(px)


# # to modify the pixel values
# img[100,100] = [220,220,220]
# print(img[100,100])

#access image properties

# img_file = 'test_image.png'

# img = cv2.imread(img_file, cv2.IMREAD_COLOR)
# alpha_img = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)
# grey_img= cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)

# print('RGB shape: ',img.shape)
# print('RGB shape: ',alpha_img.shape)
# print('RGB shape: ',grey_img.shape)

# #data type 
# print('image datatype', img.dtype)

# #size
# print('image size:' , img.size)

#setting region of the image

# img_raw = cv2.imread(img_file)

# roi = cv2.selectROI(img_raw)
# print(roi)

# # cropping selected ROI from the raw image
# roi_cropped=img_raw[int(roi[1]): int(roi[1]+roi[3]), int(roi[0]) :int(roi[0] + roi[2])]
# cv2.imshow("ROI image", roi_cropped)
# cv2.imwrite("cropped.jpeg", roi_cropped)
# cv2.waitKey(0)
# cv2.destroyAllWindows

#splitting and merging image

#splitting
# img= cv2.imread(img_file)

# g,b,r = cv2.split(img)
# cv2.imshow("green part of the image", g)
# cv2.imshow("red part of the image", r)
# cv2.imshow("blue part of the image", b)

# cv2.waitKey(0)

# #merging
# img1 = cv2.merge((g,b,r))
# cv2.imshow("image after merger of the three colors", img1)
# cv2.waitKey(0)


# change the image to LAB color
# color_change = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
# cv2.imshow("changed color scheme image", color_change)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


#Blending two images

# src1 = cv2.imread('test_image.png', cv2.IMREAD_COLOR)
# src2 = cv2.imread('cropped.jpeg', cv2.IMREAD_COLOR)

# img1 = cv2.resize(src1, (800,600))
# img2 = cv2.resize(src2, (800,600))

# blended_img = cv2.addWeighted(img1, 0.5, img2, 0.5, 1)
# cv2.imshow("Blended image", blended_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# applying filters on the image
# img = cv2.imread('test_image.png')
# K_sharped = np.array([[-1,-1,-1],
#                     [-1,9,-1],
#                     [-1,-1,-1]])

# sharpened = cv2.filter2D(img, -1,K_sharped)

# cv2.imshow(" Orginal image", img)
# cv2.imshow(" Filtered  image", sharpened)
# cv2. waitKey(0)
# cv2.destroyAllWindows()


# thresholding
# img = cv2.imread('test_image.png', cv2.IMREAD_GRAYSCALE)
# ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
# cannyimage = cv2.Canny(img, 50, 100)

# cv2.imshow(" Orginal image", img)
# cv2.imshow(" threshold  image", thresh)
# cv2.imshow(" canny  image", cannyimage)
# cv2. waitKey(0)
# cv2.destroyAllWindows()

# contour detection

# img = cv2.imread('test_image.png')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# #setting THRESHOLD of the gray scale function
# _, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# # Contour using findcontour functions

# contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# i = 0
# for contour in contours:
#     if i ==0:
#         i=1
#         continue
#     approx= cv2.approxPolyDP(contour, 0.01*cv2.arcLength(contour, True), True)
#     cv2.drawContours(img, [contour], 0, (255,0,255), 5)

#     #finding the center of the different  shapes
#     M = cv2.moments(contour)
#     if M['m00'] != 0.0:
#         x=int( M['m10']/M['m00'])
#         y=int( M['m01']/M['m00'])

#     # i want to put names of the shapes inside the corresponding shapes

#     if len(approx) ==3:
#         cv2.putText(img , 'Triangle', (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

#     elif len(approx) ==4:
#         cv2.putText(img , 'Quadrilateral', (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
#     elif len(approx) ==5:
#         cv2.putText(img , 'Pentagon', (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
#     elif len(approx) ==6:
#         cv2.putText(img , 'Hexagon', (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
#     else :
#         cv2.putText(img , 'Circle', (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

# # displaying things

# cv2.imshow('shapes', img) 
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#Color detection
# img = cv2.imread('test_image.png')
# # HSV hue saturation and value , hsv is commonly used in color and paint softwares
# hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# lower_blue = np.array([0,50,50])
# upper_blue= np.array([130,255,255])

# # thresold the HSV image to get only blue colors
# mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
# res = cv2.bitwise_and(img, img, mask=mask_blue)
# cv2.imshow('res', res)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


#object replacing in 2D image 
# img = cv2.imread("test_image.png", cv2.IMREAD_COLOR)

# img1 = img.copy()
# mask = np.zeros((100,300,3))
# print(mask.shape)

# pos = (100,100)
# var = img1[100:(100+mask.shape[0]), 100:(100+mask.shape[1])]= mask
# cv2.imshow("colored", img1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

  



