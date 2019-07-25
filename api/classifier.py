import pickle
import face_recognition
import numpy as np

def keystroke_classifier(req_json):
    # TODO: Implementar
    classifier = load_keystroke_model()

    pred = classifier.predict(keystroke_array)

    if pred[0] == 0:
        flag = 'False'
    elif pred[0] == 1:
        flag = 'True'

    return flag

def face_classifier(req_image):
    pic = face_recognition.load_image_file('/app/api/photos/matheus.jpg')
    registered = face_recognition.face_encodings(pic)
    
    unknown = face_recognition.load_image_file(req_image)
    encoding_unknown = face_recognition.face_encodings(unknown)[0]
    
    results = face_recognition.compare_faces(registered,encoding_unknown, tolerance=0.6)
    print(results)

    return str(results).replace('[','').replace(']','')

#TODO enviar as funções de load, pickle e ml-models para o handlers

# Keystroke
def load_keystroke_model():
    model = pickle.load(open('./bin_models/v1/model_keystroke.pickle', 'rb'))

    return model