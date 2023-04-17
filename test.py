import cv2
import numpy as np
import requests
import json
# Capture frames from a video file
cap = cv2.VideoCapture('D:/Acads/Lecture Recordings/2023-04-14 16-48-55.mp4')
frames = []

# Loop through the frames and store them as byte strings
while True:
    ret, frame = cap.read()
    if not ret:
        break
    if len(frames) < 60:
        frames.append(frame)

    else:
        break

chunk = np.array(frames, dtype=np.float32)
print(chunk)


