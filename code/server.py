#!/usr/bin/python3

import os
# set caffe logging to warnings & errors only
os.environ['GLOG_minloglevel'] = '2'

from flask import Flask, request, jsonify, send_from_directory
from io import BytesIO
from PIL import Image
import base64
import numpy as np
import caffe
from sklearn.neighbors import NearestNeighbors

def load_model(model_root):
    pretrained_model = model_root + 'final.caffemodel' 
    sketch_proto = model_root + 'sketchdeploy.prototxt'
    model = caffe.Net(sketch_proto, pretrained_model, caffe.TEST)
    # seems a little strange to have a transform on the sketch images?
    transformer = caffe.io.Transformer({'data': np.shape(model.blobs['data'].data)})
    transformer.set_mean('data', np.array([104, 117, 123]))
    transformer.set_transpose('data',(2,0,1))
    transformer.set_channel_swap('data', (2,1,0))
    transformer.set_raw_scale('data', 255.0)
    return model, transformer

def load_datasets(datasets_root, n_neighbors=10):
    datasets = {}
    dataset_names = next(os.walk(datasets_root))[1]
    for dataset_name in dataset_names:
        print('Loading dataset ' + dataset_name)
        filenames_path = os.path.join(datasets_root, dataset_name, 'filenames.txt')
        features_path = os.path.join(datasets_root, dataset_name, 'features.npy')
        filenames = open(filenames_path).read().splitlines()
        features = np.load(features_path)
        neighbors = NearestNeighbors(n_neighbors=n_neighbors, algorithm='brute', metric='cosine').fit(features)
        datasets[dataset_name] = {
            'filenames': filenames,
            'features': features,
            'neighbors': neighbors
        }
    return datasets

def get_features(transformer, model, img):
    sketch = [transformer.preprocess('data', img)]
    sketch = np.asarray(sketch)
    results = model.forward(data=sketch)
    features = results['pool5/7x7_s1_s'].reshape(-1)
    return features

def get_knn_filenames(dataset, features):
    features = features.reshape(1, -1)
    knns = dataset['neighbors'].kneighbors(features)
    indices = knns[1].reshape(-1)
    filenames = [dataset['filenames'][i] for i in indices]
    return filenames

def base64_png_to_numpy(data):
    return np.array(Image.open(BytesIO(base64.b64decode(data))))

caffe.set_mode_gpu()
model, transformer = load_model('../models/triplet_googlenet/Triplet_googlenet_')
datasets = load_datasets('../../datasets')

app = Flask(__name__)

@app.route('/sketchy')
def root():
    return app.send_static_file('index.html')

@app.route('/sketchy/download/<dataset>/<image>')
def send_js(dataset, image):
    root = os.path.join('static', 'datasets', dataset, 'images')
    return send_from_directory(root, image)

@app.route('/sketchy/upload/<dataset>', methods=['POST'])
def upload(dataset):
    img = base64_png_to_numpy(request.data)
    img = img.astype(float) / 255
    features = get_features(transformer, model, img)
    filenames = get_knn_filenames(datasets[dataset], features)
    return jsonify(filenames)

app.run(host='0.0.0.0', port=5000)
