import pickle

import face_recognition

import array as arr

#import handlers.pickle_handler as model_handler
import handlers.data_transform as data_transform

def keystroke_classifier(req_json):
    # TODO: Implementar
    classifier = load_keystroke_model()

    keystroke_array = data_transform.array_for_predict(req_json)

    pred = classifier.predict(keystroke_array)

    return 1


def face_classifier(req_image):
    # TODO: Implementar
    #classifier = load_facial_model()
    pic = face_recognition.load_image_file(req_image)
    req_encoding = face_recognition.face_encodings(pic)[0]
    
    registered_encoding = [arr.array([ 0.00789261,  0.10076763,  0.03704027, -0.15578765, -0.02719323,
        -0.04145699, -0.02467304, -0.00927421,  0.14241332, -0.09669774,
         0.14749482, -0.0207377 , -0.21751408,  0.03125988, -0.0363412 ,
         0.06486715, -0.18549238, -0.15221725, -0.07997507, -0.14490964,
         0.09736492,  0.04462625,  0.00465853,  0.11697571, -0.13510436,
        -0.19771251, -0.06607475, -0.11411479,  0.10554749, -0.12615523,
         0.0270316 ,  0.02724688, -0.1566442 ,  0.0305061 ,  0.01175963,
         0.00359656, -0.02282058, -0.11687636,  0.20910466,  0.04271177,
        -0.19261454,  0.02125335, -0.03031529,  0.31350899,  0.19402677,
         0.03293198,  0.01473952, -0.01874426,  0.08323692, -0.24557191,
         0.06254423,  0.2189074 ,  0.07421904,  0.04938648,  0.0670715 ,
        -0.20472586,  0.03582143,  0.11390416, -0.19788536,  0.09077469,
         0.02284055, -0.09643573,  0.02717954, -0.09425359,  0.10542111,
         0.04039697, -0.1852961 , -0.09719797,  0.10925221, -0.18890493,
        -0.04382575,  0.11405366, -0.11523133, -0.1719745 , -0.21163239,
         0.01133748,  0.4032971 ,  0.22076324, -0.11830848,  0.06248952,
        -0.05971726,  0.00344248,  0.07310766,  0.10681957, -0.14236641,
         0.02729341, -0.05913095,  0.089451  ,  0.1129452 ,  0.00121734,
        -0.11322197,  0.17989457, -0.06444252,  0.00177895, -0.01819644,
         0.08640455, -0.10074909,  0.02202147, -0.04331008,  0.0571409 ,
         0.00940124, -0.13684565, -0.04963195,  0.10476884, -0.18739837,
         0.14263695, -0.00436025, -0.04243648, -0.10183478,  0.0787111 ,
        -0.11243583,  0.01901347,  0.14461331, -0.31464952,  0.16125704,
         0.28536263,  0.03204338,  0.14260048,  0.08111482,  0.02137701,
         0.03877062, -0.10322326, -0.16144559, -0.12704827,  0.01981606,
         0.02781468,  0.09876831,  0.00985027])]
    
    results = face_recognition.compare_faces(registered_encoding, req_encoding,tolerance=0.6)
    
    return max(results)

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