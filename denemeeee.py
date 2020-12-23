
"""
Created on Sun Dec 20 23:35:05 2020

@author: vefak
"""

import numpy as np
import cv2
import math
import matplotlib.pyplot as plt


def dot(vA, vB):
    return vA[0]*vB[0]+vA[1]*vB[1]
def ang(lineA, lineB):
    # Get nicer vector form
    vA = [(lineA[0][0]-lineA[1][0]), (lineA[0][1]-lineA[1][1])]
    vB = [(lineB[0][0]-lineB[1][0]), (lineB[0][1]-lineB[1][1])]
    # Get dot prod
    dot_prod = dot(vA, vB)
    # Get magnitudes
    magA = dot(vA, vA)**0.5
    magB = dot(vB, vB)**0.5
    # Get angle in radians and then convert to degrees
    angle = math.acos(dot_prod/magB/magA)
    # Basically doing angle <- angle mod 360
    ang_deg = math.degrees(angle)%360

    if ang_deg-180>=0:
        # As in if statement
        return 360 - ang_deg
    else: 

        return ang_deg

    

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
    
def shapeDetermine(angs,contour):    
    uerr = 1.05
    derr = 0.95
    
    if  any(x > 60*derr and x < 60*uerr for x in angs):
        cv2.drawContours(img, [contour], 0, (255, 255, 0), -1)
        
    elif  any(x > 90*derr and x < 90*uerr for x in angs):
        cv2.drawContours(img, [contour], 0, (0, 0, 200), -1)
    
    elif  any(x > 108*derr and x < 108*uerr for x in angs):
        cv2.drawContours(img, [contour], 0, (255,105,180), -1)
        
    elif  any(x > 120*derr and x < 120*uerr for x in angs):
        cv2.drawContours(img, [contour], 0,(255,69,0), -1)
    else:
        cv2.drawContours(img, [contour], 0,(0,200,0), -1)
           
    
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
img2 = cv2.imread("./shapes.bmp")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (7,17),0)
_, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
_, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

filtered = []
for c in contours:
	if cv2.contourArea(c) <50:continue
	filtered.append(c)





numberedges = []
for j, cnt in enumerate(filtered):
    angles = []
    p = cv2.arcLength(cnt,True)
    res = RDP(cnt[:,0,:],p*0.01)    
    numberedges.append(res)
    for i in range(len(res)-2):
        line1 = (res[i], res[i+1])
        line2 = (res[i+2], res[i+1])
        angles.append(ang(line1,line2))
        cv2.putText(img,"{}".format(j),(res[0][0],res[0][1]),cv2.FONT_HERSHEY_TRIPLEX,1,(200,200,200),3)
    shapeDetermine(angles,cnt)
    j +=1  
plt.imshow(img)








