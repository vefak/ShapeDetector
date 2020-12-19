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
img  = cv2.imread("./shape.png")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)


_,contours, hierachy = cv2.findContours(binary, cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)


for cnt in contours:
    epsilon = 0.01*cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)

    cv2.drawContours(img, [approx], 0, (0,0,120),-1)
    

plt.imshow(img)
cv2.imwrite("./aa.png",img)




def RDP(line, epsilon):
    startIdx = 0
    endIdx = len(line)-1
    maxDist = 0 #var to store furthest point dist
    maxId = 0 #var to store furthest point index
     
    for i in range(1,endIdx):
        d = perpendicular_distance(line[i], line[startIdx], line[endIdx])
        
        print(line[i], line[startIdx], line[endIdx], "\n")
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
results  = RDP(contours[0][:,0,:], 9.68)
results  = RDP(line,1.0)


img  = cv2.imread("./shape.png")

#img  = cv2.imread("/home/vefak/Desktop/Intenseye_CV_Task/shapes.bmp")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

dst = cv2.Canny(binary, 50, 200, None, 3)

lines = cv2.HoughLines(dst, 1, np.pi / 10, 50, None, 0, 0)
for rho,theta in lines[:,0,:]:
    a = np.cos(theta)
    b = np.sin(theta)
    print("A= ",a)
    print("B= ",b, "\n")
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))

    cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

plt.imshow(dst)
plt.imshow(img)


