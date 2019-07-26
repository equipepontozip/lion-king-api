import pickle
import face_recognition
import numpy as np
import os
from sklearn.feature_extraction import FeatureHasher

#import handlers.pickle_handler as model_handler
import handlers.data_transform as data_transform
import sys

# navega para a pasta deste script
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

def keystroke_classifier(df):
    classifier = load_keystroke_model()

    pred = classifier.predict(df)

    return pred

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
    model = pickle.load(open('/api/api/bin_models/v1/voting.pickle', 'rb'))

    return 
    
def anomaly_classifier(ip, cpf):

    # for testing temporary purpose, test json sent:
    # {"cpf":"21079979177","ip":"192.168.5.21"} - anomaly
    # {"cpf":"34894024407","ip":"192.31.221.248"} - normal

    ip_login_database = {'192.168.5.21':20,'192.31.221.248':1}
    cpf_login_database = {'21079979177':15,'34894024407':1}
    anomaly_map = {1:'normal',-1:'anomaly'}

    # 'checks database' for velocity and location inspection 
    qtd_access_ip_last_days = ip_login_database[ip]
    qtd_access_cpf_last_days = cpf_login_database[cpf]

    loaded_model = pickle.load(open('bin_models/v1/login-LocalOutlierFactor.pickle', 'rb'))

    data = transform_data(ip,cpf,qtd_access_ip_last_days,qtd_access_cpf_last_days)

    pred = loaded_model.predict([data])

    return anomaly_map[pred[0]]


def transform_data(ip,cpf,qtd_access_ip_last_days,qtd_access_cpf_last_days):
    groups = ip.split( "." )
    equalize_group_length = "".join( map( lambda group: group.zfill(3), groups ))
    h = FeatureHasher(n_features=10, input_type='string')
    
    ip = h.transform([equalize_group_length]).toarray()[0]
    cpf = h.transform([cpf]).toarray()[0]
    
    data = np.concatenate((ip, cpf))
    data = list(data)
    
    data.append(qtd_access_ip_last_days)
    data.append(qtd_access_cpf_last_days)
    
    return data
