import numpy as np
import cv2

from flask import Flask
from flask import jsonify
from flask import request
from flask_cors import CORS, cross_origin

from flasgger import Swagger
from flasgger import swag_from
from swagger.swagger_config import swagger_configuration

from classifier import keystroke_classifier
from classifier import face_classifier
from classifier import anomaly_classifier

import json

app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
swagger = Swagger(app, config=swagger_configuration)


def decode_image(file):
    img_str = file.stream.read()
    file.close()

    nparray = np.fromstring(img_str, np.uint8)
    img_np = cv2.imdecode(nparray, cv2.IMREAD_COLOR)

    return img_np


@app.route('/', methods=['GET'])
@app.route('/status', methods=['GET'])
@cross_origin
@swag_from('swagger/status.yml')
def status():
    return jsonify({'status': 'ok'})


@app.route('/keystroke', methods=['POST'])
@cross_origin
@swag_from('swagger/keystroke.yml')
def keystroke():
    req_dict = request.get_json()

    classification = keystroke_classifier(req_dict)

    return jsonify({'classification': classification})


@app.route('/face', methods=['POST'])
@cross_origin
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

#classificação textual para remover depois

@app.route('/classify', methods=['POST'])
@swag_from('swagger/classify.yml')
def route_classify():
    req_dict = request.get_json()

    try:
        classification = text_classify(req_dict['text'])
    except KeyError:
        return jsonify({'Error': 'Corpo da requisição inválido'}), 400

    return jsonify({'classification': classification})

@app.route('/anomaly', methods=['POST'])
def anomaly_classify():
    ip = request.get_json()['ip']
    cpf = request.get_json()['cpf']
    classification = anomaly_classifier(ip,cpf)
    
    return classification
