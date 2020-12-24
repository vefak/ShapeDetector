#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 17:15:19 2020
@author: vefak

It takes the image from local path and go through shape detection phase:
1. Do image preprocessing
2. Find Contours
3. Use Ramer–Douglas–Peucker to reduce number of points.
4. Generate lines from points.
5. Find angles between them.
6. Coloring contours by angle value.
"""

from shape_detector import ShapeDetector
import cv2

def main():
    
    image  = cv2.imread("./shapes.bmp")
    upperlimit = 1.05
    lowerlimit = 0.95
    s_detector = ShapeDetector(image, upperlimit, lowerlimit)
    s_detector.detect_shape()
    s_detector.show_image()
    s_detector.save_image("./results.jpg")
    
if __name__ == "__main__":
    main()