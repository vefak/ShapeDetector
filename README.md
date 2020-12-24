# **Shape Detector** 

_Vefak Murat Akman_

---

It takes the image from local path and go through shape detection phase:
1. Do image preprocessing
2. Find Contours
3. Use Ramer–Douglas–Peucker to reduce number of points.
4. Generate lines from points.
5. Find angles between them.
6. Coloring contours by angle value.


**Important Note**
OpenCV Library Version must be 3.4.3


[//]: # (Image References)

[image1]: ./img/shapes.bmp "Shapes"
[image2]: ./img/results.jpg "Output"
[image3]: ./img/distance.png "Perpendicular Distance" 
[image4]: ./img/angle_formula.png "Angle Between Two Lines" 

__**Original Image**__   

 ![alt text][image1]  
 
__**Resulting Image**__   

 ![alt text][image2]  


### **Additional Notes** 

__**Distance Formula**__   
![alt text][image3]  

__**Angle Between Two Lines**__   
![alt text][image4]  

### **References** 
- [Ramer–Douglas–Peucker](https://en.wikipedia.org/wiki/Ramer%E2%80%93Douglas%E2%80%93Peucker_algorithm)
- [Perpendicular Distance](https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line#Line_defined_by_two_points)
- [Angle Between Two Formula](https://math.stackexchange.com/questions/1269050/finding-the-angle-between-two-line-equations/1269620)