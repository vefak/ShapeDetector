#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 14:45:04 2020

@author: vefak
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math

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