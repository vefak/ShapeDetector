#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: vefak
"""
import numpy as np
from matplotlib import pyplot as plt
from skimage import io
import sys
import cv2
import math
x=10000
sys.setrecursionlimit(1000*x)
print("Recursion limiti {} e ayarlandı".format(sys.getrecursionlimit()))



img  = cv2.imread("./shapes.bmp")

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#_, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
#blur = cv2.GaussianBlur(binary,(5,5),0)




rows,cols = gray.shape
resim = np.zeros((rows,cols), dtype = 'uint8')
etet = np.zeros((rows,cols), dtype = 'uint8')

#Eşikleme
for k in range(rows):
    for l in range(cols):
        if(gray[k,l]>100):
            gray[k,l]=1 
			# 
			# 
        else:
            gray[k,l]=0
            

"""
#8 komşu
def neighbors(L,P):
    res = []
    
    if(L+1 ==1):
        res.append([L+1,P])#east
    if(P+1 ==1):
        res.append([L,P+1])#south
    if(L-1 > -1):
        res.append([L-1,P])#west   
    if(P-1 > -1):
        res.append([L,P-1])#north
    if(L+1 < rows-1 and P+1 <cols-1):#+1+1
        res.append([L+1,P+1])
    if(L+1<rows-1 and P-1 >-1): #+1-1
        res.append([L+1,P-1])
    if(L-1>-1 and P+1<cols): #-1+1
        res.append([L-1,P+1])
    if(L-1>-1 and P-1): #-1,-1
        res.append([L-1,P-1])
      
    return res
"""
#4 komşu
def neighbors(L,P):
    res = []
    if(L+1 < rows):
        res.append([L+1,P])#east
    if(P+1 < cols):
        res.append([L,P+1])#south
    if(L-1 > -1):
        res.append([L-1,P])#west   
    if(P-1 > -1):
        res.append([L,P-1])#north
    
    return res   

def search(LB,label,i,j):
    LB[i,j]=label
    Nset=neighbors(i,j)
    for m in range(len(Nset)):
        a,b = Nset[m]
        if(LB[a,b]==-1):
            search(LB,label,a,b)
    

     
def find_components(LB,label):
    for i in range(rows):
        for j in range(cols):
            if(LB[i,j]==-1):
                komsular=neighbors(i,j)
                komsuda_etiket_var= False
                for n in range(0,len(komsular)):
                    a,b = komsular[n]
                    if (LB[a,b]>0):
                        label=LB[komsular[n]]
                        komsuda_etiket_var = True
                if not komsuda_etiket_var:
                    label = label+1
                    search(LB, label,i,j)
                        
 
def recursive_connected(B,LB):
    LB= B*-1 
    label=0
    find_components(LB,label)
    etet = LB
    plt.imshow(etet)
    return etet
    

filt = np.zeros((rows,cols), dtype = 'uint8')     
etet=recursive_connected(gray,resim)

for i in range(rows):
        for j in range(cols):
            if etet[i][j] == 4:
                filt[i][j] = 1
            else:
                filt[i][j] = 0

plt.imshow(etet)
plt.imshow(filt)

_,im = cv2.threshold(filt, 128, 255, cv2.THRESH_BINARY)

moments = cv2.moments(filt)
huMoments = cv2.HuMoments(moments)
# Log scale hu moments
for i in range(0,7):
  huMoments[i] = -1* math.copysign(1.0, huMoments[i]) * math.log10(abs(huMoments[i]))