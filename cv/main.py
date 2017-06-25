import http.client, urllib.request, urllib.parse, urllib.error, base64, sys

import cv2
import sys, random
import numpy as np
import requests
#import png
import copy
import time
import base64
import pandas as pd
import json
import cloudinary

from cloudinary import uploader


font = cv2.FONT_HERSHEY_SIMPLEX
emotions = ['disgust', 'surprise', 'anger', 'sadness', 'happiness', 'neutral', 'fear', 'contempt' ]

faceCascade = cv2.CascadeClassifier('haarcascades_cuda/haarcascade_frontalface_default.xml')

video_capture = cv2.VideoCapture(0)

cloudinary.config( 
  cloud_name = "skysla", 
  api_key = "959818257589146", 
  api_secret = "DC6jM0L_tjYEcUXM3NYJgnh_qrw" 
)

EMOJI_DIMENSION = 512
CAPTURE_INTERVAL = 10
IMGUR_API_KEY='f2de9155b89b3f4'
CLOSENESS_THRESHOLD = 1000*1000

class Emoji(object):
    def __init__(self, filename):
        #import ipdb; ipdb.set_trace()
        self.image = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        #self.image = cv2.resize(image, (EMOJI_DIMENSION, EMOJI_DIMENSION), interpolation = cv2.INTER_AREA)
        self.alpha_channel = self.image[:,:,3].astype(np.float32) / 255.0
        self.rgb_channel = self.image[:,:,:3].astype(np.float32)

    def blend_image(self, frame, x, y, width, height):
        alpha_channel = cv2.resize(self.alpha_channel, (height, width))
        rgb_channel = cv2.resize(self.rgb_channel, (height, width))
        origin_img = frame[y:y+height, x:x+width].astype(np.float32)
        front_img = np.stack([cv2.multiply(rgb_channel[:,:,channel_idx], alpha_channel) for channel_idx in range(3)], axis=2)
        back_img = np.stack([cv2.multiply(origin_img[:,:,channel_idx], 1-alpha_channel) for channel_idx in range(3)], axis=2)
        frame[y:y+height, x:x+width] = (front_img + back_img).astype(np.int)

emojis = {}
for emotion in emotions:
    filename = "../Emoji/" + emotion + '.png'
    emojis[emotion] = Emoji(filename)

    
def capture_image(frame):
    new_frame = copy.deepcopy(frame)
    # new_frame[:,:,0] = frame[:,:,2]
    # new_frame[:,:,2] = frame[:,:,0]

    #ss = png.from_array(new_frame, 'RGB')
    filename = '%d.jpg' % int(time.time())
    cv2.imwrite(filename, new_frame)
    #ss.save(filename)
    return filename

    
def upload_image(filename):  
    url_json = uploader.upload(filename)
    url = url_json['url']
    print(url)
    return url
        
def capture_and_upload_image(frame):
    filename = capture_image(frame)
    return upload_image(filename)


def feedImageURL(url_link):
    headers = {
        # Request headers. Replace the placeholder key below with your subscription key.
        'Content-Type': 'application/json',
        'Ocp-Apim-Subscription-Key': '5fd2f9f7fbbc4f448fbda4e2b4ad6cf0',
    }
    params = urllib.parse.urlencode({
    })
    # Replace the example URL below with the URL of the image you want to analyze.
    body = str({ 'url': str(url_link)  })
    try:
        conn = http.client.HTTPSConnection('westus.api.cognitive.microsoft.com')
        conn.request("POST", "/emotion/v1.0/recognize?%s" % params, body, headers)
        response = conn.getresponse()
        data = response.read()
        conn.close()
        return data
    except Exception as e:
        print(e.args)


def findScores(image_json):
    face_timeline = pd.DataFrame(columns = {'disgust', 'surprise', 'anger', 'sadness', 'happiness', 'neutral', 'fear', 'contempt', 'height', 'width', 'left', 'top'})
    image_json = image_json.decode()
    image_json = json.loads(image_json)
    num_faces = len(image_json)
    for idx in range(num_faces):
        emotion_score_values = image_json[idx]['scores']
        face_dimensions = image_json[idx]['faceRectangle']
        for key in face_dimensions.keys():
            emotion_score_values[key] = face_dimensions[key]
        face_timeline = face_timeline.append(emotion_score_values, ignore_index = True)
    return face_timeline


def returnEmotionDimensions(face_timeline):
    num_faces = len(face_timeline)
    df_emotions = face_timeline[['disgust', 'surprise', 'anger', 'sadness', 'happiness', 'neutral', 'fear', 'contempt']]
    top_emotion = df_emotions.idxmax(axis = 1)
    emotion_dimension_list = []
    for num in range(num_faces):
        emotion_dimension_json = {}
        best_emotion = top_emotion[num]
        emotion_dimension_json['emotion'] = best_emotion
        emotion_dimension_json['top'] = face_timeline.loc[num]['top']
        emotion_dimension_json['width'] = face_timeline.loc[num]['width']
        emotion_dimension_json['height'] = face_timeline.loc[num]['height']
        emotion_dimension_json['left'] = face_timeline.loc[num]['left']
        emotion_dimension_list.append(emotion_dimension_json)
    return emotion_dimension_list


class Face(object):
    face_id = 0
    face_id_displayed = {}
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.emotion = 'neutral'
        self.face_id = Face.face_id
        Face.face_id += 1


    def get_center(self):
        return (self.x + self.w / 2.0, self.y + self.h / 2.0)

def distance_squared_between(face1, face2):
    x1, y1 = face1.get_center()
    x2, y2 = face2.get_center()
    return (x1 - x2) ** 2 + (y1 - y2) ** 2

def earth_mover(previous_frame_faces, current_frame_faces):
    edges = {}
    for a in previous_frame_faces:
        for b in current_frame_faces:
            edges[distance_squared_between(a, b)] = (a, b)
    while current_frame_faces and edges:
        min_dis = min(edges.keys()) 
        if min_dis < CLOSENESS_THRESHOLD:
            prev, current = edges[min_dis]
            current.face_id = prev.face_id
            current.emotion = prev.emotion
            new_edges = {}
            for dis, edge in edges.items():
                if not (edge[0] in (prev, current) or edge[1] in (prev, current)):
                    new_edges[dis] = edge
            edges = new_edges
        else:
            break

has_captured = False
previous_frame_faces = []
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    if not has_captured:
      has_captured = True
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        #flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    if int(time.time()) % CAPTURE_INTERVAL == 0:
        current_emotion_faces = []
        #capture_image(frame)
        url = capture_and_upload_image(frame)


        #import ipdb; ipdb.set_trace()
        azure_data = feedImageURL(url)
        face_scores = findScores(azure_data)
        emotion_dimensions_list = returnEmotionDimensions(face_scores)
        for face in emotion_dimensions_list:
            x, y, w, h = face['left'], face['top'] - face['height'], face['width'], face['height']
            f = Face(x,y,w,h)
            f.emotion = face['emotion']
            current_emotion_faces.append(f)

        earth_mover(current_emotion_faces, current_frame_faces)
   # Draw a rectangle around the faces
    current_frame_faces = []
    for (x, y, w, h) in faces:
        if w * h < 2500:
            continue
        f = Face(x, y, w, h)
        current_frame_faces.append(f)

    earth_mover(previous_frame_faces, current_frame_faces)
    for face in current_frame_faces:
        # capture image every 5 seconds
        # send to your API
        # change emoji to display
        emojis[face.emotion].blend_image(frame, x, y, w, h)
        x, y, w, h = face.x, face.y, face.w, face.h
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, str(face.face_id), tuple(int(a) for a in face.get_center()), font, 1,(255,255,255),2,cv2.LINE_AA)
        #print(face.emotion)
        # cv2.putText(frame, 'hihi', f.get_center(),
        #     font,
        #     1,
        #     (0, 255, 0))
    #import ipdb; ipdb.set_trace()
    previous_frame_faces = current_frame_faces





   # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
