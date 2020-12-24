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
    """
    This class works as a shape classifier, based on the angles between lines
    which are extracted from contours.
    """

    def __init__(self, image, upperlimit, lowerlimit):
        
        self.image = image           # Input image
        self.upperlimit = upperlimit # Upper limit of shape angle
        self.lowerlimit = lowerlimit # Lower limit of shape angle
        self.min_pixel_area = 50     # Shapes with a minimum area of 50 pixels can be ignored if exist.
        self.perimeter_ratio = 0.01  # Perimeter ratio 
        self.shapes_dict = {         # Dictionary holds shape variables
            "triangle":  {"angle":60,  "color":(255,255,0)   },
            "rectangle": {"angle":90,  "color":(0,0,200)     },
            "pentagon":  {"angle":108, "color":(255,105,180) },
            "hexagon":   {"angle":120, "color":(255,69,0)    },
            "circle":    {"angle":360, "color":(0,255,0)     }}
        
    def show_image(self):
        plt.imshow(self.image)
        
    def save_image(self,path):
        plt.imsave(path,self.image)
        
    def __dot(self, v_a, v_b):
        """
        Calculates dot product of two vector
        """
        return v_a[0]*v_b[0]+v_a[1]*v_b[1]
    
    def __find_angle(self, lineA, lineB):
        """
        Calculates angle between two lines by formula:
        cos(θ)=(a⋅b)/(|a||b|)

        """
        # Vector form
        v_a = [(lineA[0][0]-lineA[1][0]), (lineA[0][1]-lineA[1][1])]
        v_b = [(lineB[0][0]-lineB[1][0]), (lineB[0][1]-lineB[1][1])]
        # Dot product of vectors
        dot_prod = self.__dot(v_a, v_b)
        # Get magnitudes
        mag_a = self.__dot(v_a, v_a)**0.5
        mag_b = self.__dot(v_b, v_b)**0.5
        # Get angle in radians 
        angle = math.acos(dot_prod/mag_b/mag_a)
        # Convert radian to angle
        angle_degree = math.degrees(angle)%360
        if angle_degree-180>=0:
            return 360 - angle_degree
        else: 
            return angle_degree
     
          
    def __check_contours_size(self, contours):
        """
        Check shape size bigger than 50 pixel
        """
        filtered = []
        for c in contours:
            if not cv2.contourArea(c) < self.min_pixel_area:
                filtered.append(c)
        return filtered
      
    def __get_contours(self,preprocessed_image):
        """
        Get contours of shapes in given image
        """
        new_contours = []
        _, contours, _ = cv2.findContours(preprocessed_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        new_contours = self.__check_contours_size(contours)
        return new_contours
    
    def __perpendicular_distance(self, point, point1, point2):
        """
        Calculate perpendicular distance between lines
        
        """    
        dx = point2[0] - point1[0] # (x2-x1) 
        dy = point2[1] - point1[1] # (y2-y1) 
        mag = math.sqrt(dx * dx + dy * dy) # Magnitude 
        px = point1[0]-point[0] # (x1-x0)
        py = point1[1]-point[1] # (y1-y0)   
        dp = abs(dx*py - dy*px) # |(x2-x1)(y1-y0) - (y2-y1)(x1-x0)|
        dist = dp/mag 
        return dist
    
    def __RDP(self,line,epsilon):
        """
        Using Ramiend Daquet Algorithm to reduce number of contours
        """
        start_idx = 0
        end_idx = len(line)-1
        max_dist = 0
        max_id = 0
        
        for i in range(1,end_idx):
            d = self.__perpendicular_distance(line[i],line[start_idx],line[end_idx])
            if d > max_dist:
                max_dist = d
                max_id = i
        if  max_dist < epsilon:
            results = np.vstack((line[0], line[end_idx]))
            return results
        else:
            l = self.__RDP(line[start_idx:max_id+1],epsilon)
            r = self.__RDP(line[max_id:],epsilon)
            results = np.vstack((l[:-1], r))
            return results
    
    
    def __coloring_shapes(self,angle,contour):    
        """
        Coloring shapes
        """
        uerr = self.upperlimit
        derr = self.lowerlimit
        #Rectangle
        if  angle > self.shapes_dict["rectangle"]["angle"]*derr and angle < self.shapes_dict["rectangle"]["angle"]*uerr:
            cv2.drawContours(self.image, [contour], 0, self.shapes_dict["rectangle"]["color"], -1)
        #Triangle
        elif angle > self.shapes_dict["triangle"]["angle"]*derr and angle < self.shapes_dict["triangle"]["angle"]*uerr:
            cv2.drawContours(self.image, [contour], 0, self.shapes_dict["triangle"]["color"], -1)
        #Pentagon
        elif  angle > self.shapes_dict["pentagon"]["angle"]*derr and angle < self.shapes_dict["pentagon"]["angle"]*uerr:
            cv2.drawContours(self.image, [contour], 0, self.shapes_dict["pentagon"]["color"], -1)
        #Hexagon
        elif  angle > self.shapes_dict["hexagon"]["angle"]*derr and angle < self.shapes_dict["hexagon"]["angle"]*uerr:
            cv2.drawContours(self.image, [contour], 0, self.shapes_dict["hexagon"]["color"], -1)
        #Circle
        else:
            cv2.drawContours(self.image, [contour], 0,self.shapes_dict["circle"]["color"], -1)
        
    
    
    def __convert_gray_scale(self):
        """
        Convert RGB images to Gray images
        """
        gray_image = 0.2126*self.image[:,:,0] + 0.7152*self.image[:,:,1] + 0.0722*self.image[:,:,2]
        return gray_image

        
    def __thresholding(self, gray_image):
        for k in range(len(gray_image)):
            for l in range(len(gray_image[0])):
                if(gray_image[k,l]>100):
                    gray_image[k,l]=255 
                else:
                    gray_image[k,l]=0         
        return np.uint8(gray_image)

    
    def preprocess(self):
        """
        Preprocessing given images:
            Convert gray scale
            Thresholding
        """
        gray_image = self.__convert_gray_scale()
        thresholded_image = self.__thresholding(gray_image)

        return thresholded_image
    
    def detect_shape(self):
        """
        Main function that handles all process
        """
        preprocessed_image = self.preprocess()
        contours = self.__get_contours(preprocessed_image)
        for cnt in contours:
            angles = []
            p = cv2.arcLength(cnt,True)
            res = self.__RDP(cnt[:,0,:], p*self.perimeter_ratio)    
            for i in range(len(res)-2):
                line1 = (res[i], res[i+1])
                line2 = (res[i+2], res[i+1])
                angles.append(self.__find_angle(line1,line2))
                self.__coloring_shapes(min(angles),cnt)
        