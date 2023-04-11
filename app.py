from flask import Flask, Response
from c3d_feature_extractor import getFeatureExtractor
import logging
import threading

"""Initiate Flask Object and Violence Detector"""
app = Flask(__name__)
feature_extractor = getFeatureExtractor("weights/weights.h5", "feature_extractor.h5", "fc6", False)
# svm loader
logging.info("Feature Extraction Model Loaded")


@app.get('/predict')
def predict():
    pass

