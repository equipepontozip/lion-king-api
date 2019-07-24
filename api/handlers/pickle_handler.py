def load_model():
    model = pickle.load(open('bin-models/v1/rf-classifier', 'rb'))

    return model


def load_vectorizer():
    vectorizer = pickle.load(open('bin-models/v1/tf-idf-vectorizer', 'rb'))

    return vectorizer


def vectorize_text(text):
    vectorizer = load_vectorizer()

    text_vectorized = vectorizer.transform([text])

    return text_vectorized