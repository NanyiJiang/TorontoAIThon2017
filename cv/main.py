import cv2
import sys, random
import numpy as np
import requests
import png
import copy
import time
import base64

font = cv2.FONT_HERSHEY_SIMPLEX
emotions = ['Angry', 'Contempt', 'Disgust']

faceCascade = cv2.CascadeClassifier('haarcascades_cuda/haarcascade_frontalface_default.xml')

video_capture = cv2.VideoCapture(0)

EMOJI_DIMENSION = 512
CAPTURE_INTERVAL = 5
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
    new_frame[:,:,0] = frame[:,:,2]
    new_frame[:,:,2] = frame[:,:,0]
    ss = png.from_array(new_frame, 'RGB')
    filename = '%d.png' % int(time.time())
    ss.save(filename)
    return filename
    
    
def upload_image(filename):
    with open(filename, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        res = requests.post('https://api.imgur.com/3/image', 
          headers={'Authorization': 'Client-ID %s' % IMGUR_API_KEY},
          json={'image': encoded_string})
        url = res.json()['data']['link']
        print(url)
        return url
        
def capture_and_upload_image(frame):
    filename = capture_image(frame)
    upload_image(filename)



class Face(object):
    face_id = 0
    face_id_displayed = {}
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.emotion = None
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
        # emojis[emotions[int(random.random()*3)]].blend_image(frame, x, y, w, h)
        x, y, w, h = face.x, face.y, face.w, face.h
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, str(face.face_id), tuple(int(a) for a in face.get_center()), font, 1,(255,255,255),2,cv2.LINE_AA)
        # cv2.putText(frame, 'hihi', f.get_center(),
        #     font,
        #     1,
        #     (0, 255, 0))
    #import ipdb; ipdb.set_trace()
    previous_frame_faces = current_frame_faces


    # if int(time.time()) % CAPTURE_INTERVAL == 0:
    #     capture_and_upload_image(frame)

   # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()