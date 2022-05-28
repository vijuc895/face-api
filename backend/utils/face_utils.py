from json.tool import main
import os
import io
import json
from traceback import print_tb
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
import requests
from PIL import Image, ImageDraw, ImageFont


credential = json.load(open('/Users/vijender/personalised_ad/backend/conf/AzureCloudKeys.json'))
API_KEY = credential['API_KEY']
ENDPOINT = credential['ENDPOINT']
face_client = FaceClient(ENDPOINT, CognitiveServicesCredentials(API_KEY))



def get_facial_features(image_path):
    img = open(image_path, 'rb')
    
    response_detection = face_client.face.detect_with_stream(
                            image=img,
                            detection_model='detection_01',
                            recognition_model='recognition_04',
                            return_face_attributes=['age', 'gender', 'emotion'],
                        )

    if not response_detection:
        raise Exception('No face detected')

    print(response_detection)

    facial_feature_response = {}
    face_idx = 0

    for face in response_detection:
        emotion = face.face_attributes.emotion
        emotion_dict = {"neutral": emotion.neutral * 100, "happiness": emotion.happiness * 100, "anger": emotion.anger * 100, "sadness": emotion.sadness * 100}
        emotion = max(emotion_dict, key= lambda x: emotion_dict[x])

        face_feat = {"age": face.face_attributes.age, "gender": face.face_attributes.gender, "emotion": emotion}
        face_idx += 1
        facial_feature_response["person_" + str(face_idx)] = face_feat

    return facial_feature_response

        




if __name__ == '__main__':
    
    image_path = '/Users/vijender/personalised_ad/backend/data/download.jpeg'
    
    print(get_facial_features(image_path=image_path))
