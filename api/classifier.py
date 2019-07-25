import pickle

import face_recognition

import numpy as np

#import handlers.pickle_handler as model_handler
import handlers.data_transform as data_transform

def keystroke_classifier(req_json):
    # TODO: Implementar
    classifier = load_keystroke_model()

    keystroke_array = data_transform.array_for_predict(req_json)

    pred = classifier.predict(keystroke_array)

    return 1

def face_classifier(req_image):
    pic = face_recognition.load_image_file('./api/photos/matheus.jpg')
    registered = face_recognition.face_encodings(pic)
    
    unknown = face_recognition.load_image_file(req_image)
    encoding_unknown = face_recognition.face_encodings(unknown)[0]
    
    results = face_recognition.compare_faces(registered,encoding_unknown, tolerance=0.6)
    print(results)

    return str(results).replace('[','').replace(']','')

def text_classify(text):
    text_vectorized = vectorize_text(text)
    classifier = load_text_model()

    pred = classifier.predict(text_vectorized)

    flag = None

    if pred[0] == 0:
        flag = 'Red flag'
    elif pred[0] == 1:
        flag = 'Green flag'

    return flag
    


#TODO enviar funções para o handlers

# Keystroke
def load_keystroke_model():
    model = pickle.load(open('./bin_models/v1/model_keystroke.pickle', 'rb'))

    return model


# Facial
def load_facial_model():
    model = pickle.load(open('./bin_models/v1/model_facial.pickle', 'rb'))

    return model


# Text
def load_text_model():
    model = pickle.load(open('./bin_models/v1/rf-classifier', 'rb'))

    return model

def load_text_vectorizer():
    vectorizer = pickle.load(open('./bin_models/v1/tf-idf-vectorizer', 'rb'))

    return vectorizer

def vectorize_text(text):
    vectorizer = load_text_vectorizer()

    text_vectorized = vectorizer.transform([text])

    return text_vectorized