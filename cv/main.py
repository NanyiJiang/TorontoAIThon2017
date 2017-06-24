import cv2
import sys, random
import numpy as np

emotions = ['Angry', 'Contempt', 'Disgust']

faceCascade = cv2.CascadeClassifier('haarcascades_cuda/haarcascade_frontalface_default.xml')

video_capture = cv2.VideoCapture(0)

EMOJI_DIMENSION = 512

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

#s_img = cv2.cvtColor(s_img, cv2.COLOR_BGR2GRAY)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        #flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )

   # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        # capture image every 5 seconds
        # send to your API
        # change emoji to display
        emojis[emotions[int(random.random()*3)]].blend_image(frame, x, y, w, h)
    # origin_img = frame[y:y+height, x:x+width].astype(np.float32)
    # front_img = np.stack([cv2.multiply(emoji_rgb_channel[:,:,channel_idx], emoji_alpha_channel) for channel_idx in range(3)], axis=2)
    # back_img = np.stack([cv2.multiply(origin_img[:,:,channel_idx], 1-emoji_alpha_channel) for channel_idx in range(3)], axis=2)
    # frame[y:y+height, x:x+width] = (front_img + back_img).astype(np.int)
    # = emoji_alpha_channel * frame[y:y+height, x:x+width] + (1-emoji_alpha_channel) * emoji_rgb_channel

   # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()