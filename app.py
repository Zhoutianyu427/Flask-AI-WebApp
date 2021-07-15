from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import tensorflow as tf


# Import Azure Image Analysis Service (Add from Tianyu)
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials

cog_key = '2f07ed8b7d6f418db7f9e3e70da1e5b2'
cog_endpoint = 'https://cognitivtianyuzhou.cognitiveservices.azure.com/'


# Keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50


# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
# MODEL_PATH = 'models/your_model.h5'

graph = tf.compat.v1.get_default_graph()

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x, mode='caffe')

    preds = model.predict(x)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route("/videoclassification", methods=["GET"])
def videoclassification():
	return render_template("video.html")

@app.route("/imageclassification", methods=["GET"])
def imageclassfication():
    return render_template("image.html")

@app.route("/predict", methods=["GET", "POST"])
# def imageclassification():
# 	return render_template("image.html")
def upload():
    global graph
    with graph.as_default():
        if request.method == 'POST':
            # Get the file from post request

            f = request.files['image']

            # Save the file to ./uploads
            basepath = os.path.dirname(__file__)
            file_path = os.path.join(
                basepath, 'uploads', secure_filename(f.filename))
            f.save(file_path)

            # Make prediction by using cognitive service from Azure (Tianyu)
            computervision_client = ComputerVisionClient(cog_endpoint, CognitiveServicesCredentials(cog_key))

            # Get a description from the computer vision service (Tianyu)
            image_stream = open(file_path, "rb")
            description = computervision_client.describe_image_in_stream(image_stream)
            caption_text = ''
            if (len(description.captions) == 0):
                caption_text = 'No caption detected'
            else:
                for caption in description.captions:
                    caption_text = caption_text + " '{}'\n(Confidence: {:.2f}%)".format(caption.text, caption.confidence * 100)

            return caption_text
        return None

@app.route("/movementclassification", methods=["GET"])
def movementclassification():
	return render_template("movement.html")

@app.route("/contact", methods=["GET"])
def contact():
    return render_template("contact.html")

if __name__ == '__main__':
	# app.run(port=5002, debug=True)

	# Serve the app with gevent
	http_server = WSGIServer(('0.0.0.0', 5000), app)
	http_server.serve_forever()

