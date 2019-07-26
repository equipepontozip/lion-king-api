import numpy as np
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

#import os
#import sys
#abs_path = os.path.dirname(os.path.realpath(__file__))
#abs_path = abs_path.replace("api", "job")

#sys.path.append(abs_path)

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

    classification = keystroke_classifier(data)

    return jsonify({'classification': classification[0]})


@app.route('/face', methods=['POST'])
@swag_from('swagger/face.yml')
def face_recognition():
    if not 'image' in request.files:
        response = jsonify({'erro': 'Imagem não enviada no corpo da requisição'})
        response.status_code = 400
        return response

    #precisa que manda a imagem e não decodificado
    decoded_image = decode_image(request.files['image'])

    #o que é essa função decode_image???
    classification = face_classifier(decoded_image)

    return jsonify({'classification': classification})
