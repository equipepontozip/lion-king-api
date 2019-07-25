import pickle

#import handlers.pickle_handler as model_handler

def keystroke_classifier(req_json):
    # TODO: Implementar
    return 1


def face_classifier(image):
    # TODO: Implementar
    return "foo"

def load_model():
    model = pickle.load(open('/app/api/bin-models/v1/rf-classifier', 'rb'))

    return model


def load_vectorizer():
    vectorizer = pickle.load(open('./bin-models/v1/tf-idf-vectorizer', 'rb'))

    return vectorizer


def vectorize_text(text):
    vectorizer = load_vectorizer()

    text_vectorized = vectorizer.transform([text])

    return text_vectorized


def text_classify(text):
    text_vectorized = vectorize_text(text)
    classifier = load_model()

    pred = classifier.predict(text_vectorized)

    flag = None

    if pred[0] == 0:
        flag = 'Red flag'
    elif pred[0] == 1:
        flag = 'Green flag'

    return flag