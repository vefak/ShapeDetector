#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 20:30:53 2020

@author: vefak
"""


# Import Necessary library
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math

# Read Input image
img  = cv2.imread("./shapes.bmp")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(gray,(0,0),3)
sharped = cv2.(frame, 1.5,  -0.5, 0, image);

_, binary = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY)
_,contours, hierachy = cv2.findContours(blur, cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)




# It supports maximum 7x7 window!
    median_size = min(median_size, 7)
    # apply the (median_size X median_size) median filter on the image [default 5 x 5]
    processed_image = cv2.medianBlur(image, median_size)




plt.imshow(blur)
lines = []
for i, cnt in enumerate(contours[0][:,0,:], start=0):
    lines.append([cnt])
    
line_one = (lines[0],lines[1])
line_two = (lines[1],lines[2])
line_third = (lines[2],lines[3])
line_four = (lines[3],lines[0])



cv2.line(img, (76,60), (76,219), (255,0,0), 5) 
cv2.line(img, (76,219), (401,219), (0,255,0), 5) 
cv2.line(img, (401,219), (401,60) ,(0,255,255), 5) 
cv2.line(img, (401,60), (76,60) ,(255,255,0), 5) 

plt.imshow(img)


def calcAngle(lineA,lineB):
    
    y11 = lineA[0][0]
    x11 = lineA[0][0]
    y12 = lineA[1][1]
    x12 = lineA[1][0]

    y21 = lineB[0][1]
    x21 = lineB[0][0]
    y22 = lineB[1][1]
    x22 = lineB[1][0]

    #calculate angle between pairs of lines
    angle1 = math.atan2(y11-y12,x11-x12)
    angle2 = math.atan2(y21-y22,x21-x22)
    angleDegrees = (angle1-angle2) * 360 / (2*math.pi)
    
    return angleDegrees


calcAngle(line_one,line_two)
calcAngle(line_two,line_third)
calcAngle(line_third,line_four)
calcAngle(line_four,line_one)


--------------------------------------------------------------------------------

contours[2][:,0,:]
results = RDP(line, epsilon)

def RDP(line, epsilon):
    startIdx = 0
    endIdx = len(line)-1
    maxDist = 0 #var to store furthest point dist
    maxId = 0 #var to store furthest point index
     
    for i in range(1,endIdx):
        d = perpendicular_distance(line[i], line[startIdx], line[endIdx])
        
        if d > maxDist:
            maxDist = d #overwrite max distance
            maxId = i #overwrite max index
     
    if maxDist > epsilon:
     l = RDP(line[startIdx:maxId+1], epsilon)
     r = RDP(line[maxId:], epsilon)
     results = np.vstack((l[:-1], r))
     return results
     
    else:
     results = np.vstack((line[0], line[endIdx]))
     return results


def perpendicular_distance(p, p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    d = math.sqrt(dx * dx + dy * dy)
    return abs(p[0] * dy - p[1] * dx + p2[0] * p1[1] - p2[1] * p1[0])/d

line = np.array([(0,0),(1,0.1),(2,-0.1),(3,5),(4,6),(5,7),(6,8.1),(7,9),(8,9),(9,9)])
results  = RDP(line, 1.0)