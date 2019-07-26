import numpy as np
import pandas as pd
import sys
import cv2
import json

from flask import Flask
from flask import jsonify
from flask import request
from flasgger import Swagger
from flasgger import swag_from
from flask_cors import CORS

from swagger.swagger_config import swagger_configuration

from classifier import keystroke_classifier
from classifier import face_classifier
from handlers.data_transform import transform_keystroke
from classifier import anomaly_classifier

import json

app = Flask(__name__)
swagger = Swagger(app, config=swagger_configuration)
CORS(app, origins="*")


def decode_image(file):
    img_str = file.stream.read()
    file.close()

    nparray = np.fromstring(img_str, np.uint8)
    img_np = cv2.imdecode(nparray, cv2.IMREAD_COLOR)

    return img_np


@app.route('/', methods=['GET'])
@app.route('/status', methods=['GET'])
@swag_from('swagger/status.yml')
def status():
    return jsonify({'status': 'ok'})


@app.route('/keystroke', methods=['POST'])
@swag_from('swagger/keystroke.yml')
def keystroke():
    req_dict = json.loads(request.data)

    data = transform_keystroke(req_dict)

    if data == False:
        return jsonify({'classification': data})

    classification = keystroke_classifier(data)
    print("class:", classification, file=sys.stdout)

    return jsonify({'classification': classification[0]})


@app.route('/face', methods=['POST'])
@swag_from('swagger/face.yml')
def face_recognition():
    if not 'image' in request.files:
        response = jsonify({'erro': 'Imagem não enviada no corpo da requisição'})
        response.status_code = 400
        return response

    #decoded_image = decode_image(request.files['image'])
    decoded_image = request.files['image']
    
    classification = face_classifier(decoded_image)

    return jsonify({'classification': classification})

@app.route('/anomaly', methods=['POST'])
@swag_from('swagger/anomaly.yml')
def anomaly_classify():

    # for testing temporary purpose, test json sent:
    # {"cpf":"21079979177","ip":"192.168.5.21"} - anomaly
    # {"cpf":"34894024407","ip":"192.31.221.248"} - normal

    ip = request.get_json()['ip']
    cpf = request.get_json()['cpf']
    classification = anomaly_classifier(ip,cpf)
    
    return classification
