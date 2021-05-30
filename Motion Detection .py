# -*- coding: utf-8 -*-
"""
Created on Sat May 29 18:04:11 2021

@author: Microsoft
"""


import cv2 as cv 
import numpy as np
import os 
 
os.chdir("D:\Courses\Machine learning and Deep learning\open cv\Computer vision with Ahmed Ibrahim\data")

cap=cv.VideoCapture("example_2.mp4")


# (returns float which we need to convert to integer for later on!)
width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))



fourcc=cv.VideoWriter_fourcc(*'XVID')

writer=cv.VideoWriter("people_detection.avi",fourcc, 10 , (1200,700) )


ret,frame1=cap.read()
ret,frame2=cap.read()

kernal=np.ones((9,9),np.uint8)

while(1):
    
    # calculate absolute different between two frame 
    ads_diff=cv.absdiff(frame1, frame2)
    img2gray=cv.cvtColor(ads_diff, cv.COLOR_BGR2GRAY)
    #do blur to enhancement the gray image
    gaussian=cv.GaussianBlur(img2gray, (5,5),0)
    # get thresh hold tht convert image to binary 
    _,thresh=cv.threshold(gaussian, 20, 255, cv.THRESH_BINARY)
    # dilate + erode (closing) the pixels that make more accurate
    morphology=cv.morphologyEx(thresh,  cv.MORPH_CLOSE, kernal, iterations=5)
    #find the contours from the transformation image 
    contours,_=cv.findContours(morphology, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        #give bounding rectangle box from contours 
        (x,y,w,h)=cv.boundingRect(contour)
        
        if cv.contourArea(contour)<600:
            continue
        #draw rectangle on frame1
        cv.rectangle(frame1, (x,y), (x+w, y+h), (0,0,255))
        
    # cv.drawContours(frame1, contour, -1, (0,255,0),10)
    image = cv.resize(frame1, (1200,700))
    writer.write(image)
    cv.imshow("motion detection",frame1)
    cv.imshow("morphology",morphology)

            
    #give more frame and repeates
    frame1=frame2
    _,frame2=cap.read()
    
    if cv.waitKey(50)==27:
        break        

        
cap.release()
writer.release()
cv.destroyAllWindows()