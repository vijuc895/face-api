import flask
from flask import request
import base64
import uuid
from cv2 import cv2
import numpy as np
from utils.face_utils import get_facial_features

app = flask.Flask(__name__)
app.config["DEBUG"] = True

def get_image_path():
    """Creates a unique hash"""
    return str(uuid.uuid4()) + ".jpg"


@app.route('/ping')
def x():
    return 'pong'

@app.route('/face-feats', methods=['POST'])
def get_face_feats():
    post_data = request.get_json()
    image_base64 = post_data['image_base64']
    img_data = base64.b64decode (image_base64)
    image_np = np.fromstring(img_data, dtype=np.uint8)
    img = cv2.imdecode(image_np, flags=cv2.IMREAD_COLOR)
    image_path = get_image_path()

    cv2.imwrite(image_path, img)

    face_feats = get_facial_features(image_path=image_path)

    return face_feats


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8001)



    