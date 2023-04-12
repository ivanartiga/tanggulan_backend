from flask import Flask, Response, request
from c3d_feature_extractor import getFeatureExtractor
import logging
import threading
import json
import numpy as np
import cv2

"""Initiate Flask Object and Violence Detector"""
app = Flask(__name__)
feature_extractor = getFeatureExtractor("weights/weights.h5", "feature_extractor.h5", "fc6", False)
# svm loader
logging.info("Feature Extraction Model Loaded")


@app.route('/predict', methods=['POST'])
def predict():
    # Get the JSON object containing the encoded frames
    frames_json = request.get_json()
    # Decode the JSON object and extract the frames
    frames_dict = json.loads(frames_json)
    # frames array to feed into 3D CNN Feature Extractor Model
    frames = []
    for key in frames_dict:
        # Convert the encoded frame to a numpy array
        nparr = np.frombuffer(frames_dict[key], np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        frames.append(frame)
    # Preprocess Frames
    processed_frames = frame_preprocessing(frames)
    # Extract Features from Frames with 16 frame Input
    features = feature_extraction(processed_frames)
    # Feed to SVM
    # pending

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