
"""
Created on Sun Dec 20 23:35:05 2020

@author: vefak
"""

import numpy as np
import cv2
import random
import math
import matplotlib.pyplot as plt



def _perpendicular_distance(p, p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    d = math.sqrt(dx * dx + dy * dy)
    return abs((p[0] * dy) - (p[1] * dx) + (p2[0] * p1[1]) - (p2[1] * p1[0]))/d


def perpendicular_distance(p, p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    mag = math.sqrt(dx * dx + dy * dy)
    if (mag > 0.0):
        dx /= mag
        dy /= mag
    pvx = p[0]-p1[0]
    pvy = p[1]-p1[1]
    
    pvdot = dx*pvx + dy* pvy
    
    dsx = pvdot * dx
    dsy = pvdot * dy
    
    ax = pvx - dsx
    ay = pvy - dsy
    res = math.sqrt(ax*ax + ay*ay)
    return res
    
    
    

def RDP(line,epsilon):
    startIdx = 0
    endIdx = len(line)-1
    maxDist = 0
    maxId = 0
    
    for i in range(1,endIdx):
        d = perpendicular_distance(line[i],line[startIdx],line[endIdx])
        if d > maxDist:
            maxDist = d
            maxId = i
    if maxDist > epsilon:
        l = RDP(line[startIdx:maxId+1],epsilon)
        r = RDP(line[maxId:],epsilon)
        results = np.vstack((l[:-1], r))
        return results
    else:
        results = np.vstack((line[0], line[endIdx]))
        return results

img  = cv2.imread("./shapes.bmp")
plt.imshow(img)

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(gray, (3,3),0)

_, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

plt.imshow(thresh)

_, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

print(len(contours))

filtered = []

for c in contours:
	if cv2.contourArea(c) <50:continue
	filtered.append(c)

print(len(filtered))

objects = np.zeros([img.shape[0],img.shape[1],3], 'uint8')

points = []
    
for c in filtered:
    p = cv2.arcLength(c,True)
    res = RDP(c[:,0,:],p*0.01)
    points.append(res)
    color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
    for i in range(len(res)-1):
        cv2.line(img, (res[i][0],res[i][1]), (res[i+1][0],res[i+1][1]), color, 10)



plt.imshow(img)





