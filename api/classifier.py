import pickle

import face_recognition

#import handlers.pickle_handler as model_handler
import handlers.data_transform as data_transform
import sys

def keystroke_classifier(df):
    # TODO: Implementar
    classifier = load_keystroke_model()

    pred = classifier.predict(df)

    return pred


def face_classifier(req_decoded_image):
    # TODO: Implementar
    #classifier = load_facial_model()
    pic = face_recognition.load_image_file(req_decoded_image)
    encoding = face_recognition.face_encodings(pic)[0]

    results = face_recognition.compare_faces(clientes, encoding,tolerance=0.6)

    return max(results)
    #pred = classifier.predict(decoded_image)
    return "foo"


#TODO enviar funções para o handlers

# Keystroke
def load_keystroke_model():
    model = pickle.load(open('/api/api/bin_models/v1/voting.pickle', 'rb'))

    return model


# Facial
def load_facial_model():
    model = pickle.load(open('./bin_models/v1/model_facial.pickle', 'rb'))

    return model