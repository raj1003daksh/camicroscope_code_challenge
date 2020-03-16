import numpy as np 
import cv2
import matplotlib.pyplot as plt
import math


def gaussian_kernel(size, sigma=1):
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return g

def sobelfilter(image):
    gx = np.array([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])
    gy = np.array([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])
    rows, columns = np.shape(image) 
    sobel_filtered_image = np.zeros((rows, columns),np.float64)
    Gradient = np.zeros((rows, columns),np.float64) 
    for i in range(rows - 2):
        for j in range(columns - 2):
            Gx = np.multiply(gx, image[i:i + 3, j:j + 3]).sum() / 8
            Gy = np.multiply(gy, image[i:i + 3, j:j + 3]).sum() / 8
            Gradient[i+1,j+1]=np.degrees(np.arctan2(Gy,Gx))
            sobel_filtered_image[i + 1, j + 1] = np.sqrt(np.square(Gx)+np.square(Gy)) 
    sobel_filtered_image=sobel_filtered_image/np.max(sobel_filtered_image)
    return sobel_filtered_image,Gradient

def NonMaxSup(Gmag, Grad):
    NMS = np.zeros((Gmag.shape[0],Gmag.shape[1]),np.float64)
    for i in range(1, Gmag.shape[0] - 1):
        for j in range(1, Gmag.shape[1] - 1):
            if((Grad[i,j] >= -22.5 and Grad[i,j] <= 22.5) or (Grad[i,j] <= -157.5 and Grad[i,j] >= 157.5)):
                if((Gmag[i,j] > Gmag[i,j+1]) and (Gmag[i,j] > Gmag[i,j-1])):
                    NMS[i,j] = Gmag[i,j]
                else:
                    NMS[i,j] = 0
            if((Grad[i,j] >= 22.5 and Grad[i,j] <= 67.5) or (Grad[i,j] <= -112.5 and Grad[i,j] >= -157.5)):
                if((Gmag[i,j] > Gmag[i+1,j+1]) and (Gmag[i,j] > Gmag[i-1,j-1])):
                    NMS[i,j] = Gmag[i,j]
                else:
                    NMS[i,j] = 0
            if((Grad[i,j] >= 67.5 and Grad[i,j] <= 112.5) or (Grad[i,j] <= -67.5 and Grad[i,j] >= -112.5)):
                if((Gmag[i,j] > Gmag[i+1,j]) and (Gmag[i,j] > Gmag[i-1,j])):
                    NMS[i,j] = Gmag[i,j]
                else:
                    NMS[i,j] = 0
            if((Grad[i,j] >= 112.5 and Grad[i,j] <= 157.5) or (Grad[i,j] <= -22.5 and Grad[i,j] >= -67.5)):
                if((Gmag[i,j] > Gmag[i+1,j-1]) and (Gmag[i,j] > Gmag[i-1,j+1])):
                    NMS[i,j] = Gmag[i,j]
                else:
                    NMS[i,j] = 0
    return NMS

def DoThreshHyst(img):
    highThresholdRatio = 0.2
    lowThresholdRatio = 0.15
    GSup = np.copy(img)
    h = int(GSup.shape[0])
    w = int(GSup.shape[1])
    highThreshold = np.max(GSup) * highThresholdRatio
    lowThreshold = highThreshold * lowThresholdRatio   
    x = 0.1
    oldx=0
    while(oldx != x):
        oldx = x
        for i in range(1,h-1):
            for j in range(1,w-1):
                if(GSup[i,j] > highThreshold):
                    GSup[i,j] = 1
                elif(GSup[i,j] < lowThreshold):
                    GSup[i,j] = 0
                else:
                    if((GSup[i-1,j-1] > highThreshold) or 
                        (GSup[i-1,j] > highThreshold) or
                        (GSup[i-1,j+1] > highThreshold) or
                        (GSup[i,j-1] > highThreshold) or
                        (GSup[i,j+1] > highThreshold) or
                        (GSup[i+1,j-1] > highThreshold) or
                        (GSup[i+1,j] > highThreshold) or
                        (GSup[i+1,j+1] > highThreshold)):
                        GSup[i,j] = 1
        x = np.sum(GSup == 1)
    
    GSup = (GSup == 1) * GSup  
    return GSup

def canny_edges(img):
    img=cv2.GaussianBlur(img,(5,5),0)
    sobelfilterimage,gradient=sobelfilter(img)
    NMS=NonMaxSup(sobelfilterimage,gradient)
    canny=DoThreshHyst(NMS)
    return canny



img=cv2.imread('Building.jpg',0)
canny=canny_edges(img) 
cv2.imshow("cannyedge",canny)
cv2.waitKey(0)
cv2.destroyAllWindows()


