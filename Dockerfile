FROM aaftio/face_recognition:latest

WORKDIR /api

ADD requirements.txt /api/

RUN pip install -r requirements.txt

ADD . /api/

CMD flask run --host 0.0.0.0 --port 5000
