import cv2
import json
import requests
import time

cap = cv2.VideoCapture('D:/Acads/Lecture Recordings/2023-04-14 16-48-55.mp4')   # Replace 'video.mp4' with your file name or camera index

frames = []
i = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        # Store the frame in a dictionary with a key
        key = f'frame_{i}'
        frames.append((key, frame))
        i += 1
    else:
        break

# Convert the frames to encoded format and store in a dictionary
# frames_dict = {}
# for key, frame in frames:
#     # Convert the numpy array to encoded format
#     _, buffer = cv2.imencode('.jpg', frame)
#     encoded_frame = buffer.tobytes()
#     # Add the encoded frame to the dictionary with the key
#     frames_dict[key] = encoded_frame

# Convert the dictionary to a JSON object and send it to the Flask route
#frames_json = json.dumps(frames_dict)
# url = 'http://localhost:5000/predict'
# content_type = 'application/octet-stream'
# headers = {'content-type': content_type}
# response = requests.post(url,data={frames_json},headers=headers)