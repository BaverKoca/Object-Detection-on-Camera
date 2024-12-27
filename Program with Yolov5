pip install ultralytics

import cv2
import torch
import numpy as np
import time
from ultralytics import YOLO


#Download YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  #Load YOLOv5s !!!YoloV8 lastest version of Yolo but it is not available in ultralytics!!!

#Use OpenCV to open the camera
cap = cv2.VideoCapture(0)

#When the camera is opened
if not cap.isOpened():
    print("Camera didn't open.")
    exit()

#Let's capture the video stream and do object detection for 30 seconds
start_time = time.time()

while time.time() - start_time < 60:  #You can change the time over here
    ret, frame = cap.read()
    
    if not ret:
        print("Camera can't receive frame.")
        break

    #Object detection with YOLO
    results = model(frame)

    #Show the results on the screen
    results.render()  #Detected objects drawn on the frame
    cv2.imshow('YOLO Object Detection', frame)

    #You can exit by pressing the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#Close the camera and window
cap.release()
cv2.destroyAllWindows()
