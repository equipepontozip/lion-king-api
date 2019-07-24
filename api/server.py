import numpy as np
import cv2

from flask import Flask
from flask import jsonify
from flask import request
from flasgger import Swagger
from flasgger import swag_from

from swagger.swagger_config import swagger_configuration
from classifier import keystroke_classifier
from classifier import face_classifier


app = Flask(__name__)
swagger = Swagger(app, config=swagger_configuration)


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
def keystroke():
    req_dict = request.get_json()

    classification = keystroke_classifier(req_dict)

    return jsonify({'classification': classification})


@app.route('/face', methods=['POST'])
@swag_from('swagger/face.yml')
def face_recognition():
    if not 'image' in request.files:
        response = jsonify({'erro': 'Imagem não enviada no corpo da requisição'})
        response.status_code = 400
        return response

    image = decode_image(request.files['image'])
    classification = face_classifier(image)

    return jsonify({'classification': classification})
