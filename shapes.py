#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 17:15:19 2020

@author: vefak
"""


from shape_detector import ShapeDetector
import cv2
import matplotlib.pyplot as plt


def main():
    
    img  = cv2.imread("./shapes.bmp")
    s_detector = ShapeDetector()
    img = s_detector.detect_shapes(img)
    plt.imshow(img)
    
    
if __name__ == "__main__":
    main()