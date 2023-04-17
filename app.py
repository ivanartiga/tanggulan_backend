from flask import Flask, Response, request, jsonify
from c3d_feature_extractor import getFeatureExtractor
import logging
import joblib
import threading
import json
import numpy as np
import cv2


"""Initiate Flask Object and Violence Detector"""
app = Flask(__name__)
# try:
feature_extractor = getFeatureExtractor("weights/weights.h5", "feature_extractor.h5", "fc6", False)
svm = joblib.load('SVMClassifier.Model')
print("Model Loaded")
# except Exception as e:
#     print("Model Not Loaded")

labels = ["Violent","Non Violent"]

@app.route('/test', methods=['GET'])
def test():
    return "Hello User"
@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve the frames from the JSON payload
    frames = json.loads(request.form['frames'])
    processed_frames = []
    # Loop through the frames and decode them back into OpenCV frames
    for frame_bytes in frames:
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        resized = cv2.resize(frame, (112, 112))
        processed_frames.append(resized)
        # Process the frame here

    numpy_array = np.array(processed_frames, dtype=np.float32)

    return 'Frames processed successfully'



def prediction_process(frames):
    # Preprocess Frames
    processed_frames = frame_preprocessing(frames)
    # Extract Features from Frames with 16 frame Input
    features = feature_extraction(processed_frames)
    # Feed to SVM
    result = classify(features)
    print('The result of the SVM')
    print(result)

def frame_preprocessing(frames):
    processed_frames = []
    for frame in frames:
        # Example processing: convert to grayscale
        resized_frame = cv2.resize(frame, (112, 112))
        processed_frames.append(resized_frame)
    return processed_frames

def feature_extraction(input_frames):
    output = feature_extractor.predict(input_frames)
    return output

def classify(input_vectors):
    y_pred = svm.predict(input_vectors)
    return y_pred


if __name__ == '__main__':
    app.run(debug=True,threaded=True)

# Sending frames to flask server CODE

# cap = cv2.VideoCapture('video.mp4')
# frames = []
# i = 0
# while(cap.isOpened()):
#     ret, frame = cap.read()
#     if ret == True:
#         # Store the frame in a dictionary with a key
#         key = f'frame_{i}'
#         frames.append((key, frame))
#         i += 1
#     else:
#         break
#
# # Convert the frames to encoded format and store in a dictionary
# frames_dict = {}
# for key, frame in frames:
#     # Convert the numpy array to encoded format
#     _, buffer = cv2.imencode('.jpg', frame)
#     encoded_frame = buffer.tobytes()
#     # Add the encoded frame to the dictionary with the key
#     frames_dict[key] = encoded_frame
#
# # Convert the dictionary to a JSON object and send it to the Flask route
# frames_json = json.dumps(frames_dict)