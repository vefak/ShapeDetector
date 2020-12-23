#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 17:13:32 2020

@author: vefak
"""
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

class ShapeDetector:

    def __init__(self):
        
        #DrawContours
        self.upperlimit = 1.05
        self.downlimit = 0.95

        
    def __check_contours_size(self,cnts):
        filtered = []
        for c in cnts:
            if not cv2.contourArea(c) <50:
                filtered.append(c)
        return filtered
    
   
    def __dot(self, vA, vB):
        return vA[0]*vB[0]+vA[1]*vB[1]
    
    def ang(self, lineA, lineB):
        # Get nicer vector form
        vA = [(lineA[0][0]-lineA[1][0]), (lineA[0][1]-lineA[1][1])]
        vB = [(lineB[0][0]-lineB[1][0]), (lineB[0][1]-lineB[1][1])]
        # Get dot prod
        dot_prod = self.__dot(vA, vB)
        # Get magnitudes
        magA = self.__dot(vA, vA)**0.5
        magB = self.__dot(vB, vB)**0.5
        # Get angle in radians and then convert to degrees
        angle = math.acos(dot_prod/magB/magA)
        # Basically doing angle <- angle mod 360
        ang_deg = math.degrees(angle)%360
    
        if ang_deg-180>=0:
            # As in if statement
            return 360 - ang_deg
        else: 
    
            return ang_deg
    
    
    def __perpendicular_distance(self,p, p1, p2):
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
        dist = math.sqrt(ax*ax + ay*ay)
        return dist
    
    
    def get_contours(self,image):
        
        new_contours = []
        plt.imshow(image)
        _, contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        new_contours = self.__check_contours_size(contours)
    
        return new_contours
    
    def RDP(self,line,epsilon):
        startIdx = 0
        endIdx = len(line)-1
        maxDist = 0
        maxId = 0
    
        for i in range(1,endIdx):
            d = self.__perpendicular_distance(line[i],line[startIdx],line[endIdx])
            if d > maxDist:
                maxDist = d
                maxId = i
        if maxDist > epsilon:
            l = self.RDP(line[startIdx:maxId+1],epsilon)
            r = self.RDP(line[maxId:],epsilon)
            results = np.vstack((l[:-1], r))
            return results
        else:
            results = np.vstack((line[0], line[endIdx]))
            return results
    
    
    def draw_shapes(self,image,angs,contour):    
        uerr = self.upperlimit
        derr = self.downlimit
        
        if  any(x > 60*derr and x < 60*uerr for x in angs):
            cv2.drawContours(image, [contour], 0, (255, 255, 0), -1)
            
        elif  any(x > 90*derr and x < 90*uerr for x in angs):
            cv2.drawContours(image, [contour], 0, (0, 0, 200), -1)
        
        elif  any(x > 108*derr and x < 108*uerr for x in angs):
            cv2.drawContours(image, [contour], 0, (255,105,180), -1)
            
        elif  any(x > 120*derr and x < 120*uerr for x in angs):
            cv2.drawContours(image, [contour], 0,(255,69,0), -1)
        else:
            cv2.drawContours(image, [contour], 0,(0,200,0), -1)
        return image
    
    
    def __convertGrayScale(self,image):
        Y = 0.2126*image[:,:,0] + 0.7152*image[:,:,1] + 0.0722*image[:,:,2]
        return Y

        
    def __thresholding(self,image):
        
        for k in range(len(image)):
            for l in range(len(image[0])):
                if(image[k,l]>100):
                    image[k,l]=255
        			 
                else:
                    image[k,l]=0

        return np.uint8(image)

    
    def preprocess(self,image):
       
        gray = self.__convertGrayScale(image)
        thresh = self.__thresholding(gray)

        return thresh
    
    def detect_shapes(self,image):
        
        preprocessed_img = self.preprocess(image)
        filtered_contours = self.get_contours(preprocessed_img)

        for cnt in filtered_contours:
            angles = []
            p = cv2.arcLength(cnt,True)
            res = self.RDP(cnt[:,0,:],p*0.01)    
            for i in range(len(res)-2):
                line1 = (res[i], res[i+1])
                line2 = (res[i+2], res[i+1])
                angles.append(self.ang(line1,line2))
            image = self.draw_shapes(image,angles,cnt)
        return image
        