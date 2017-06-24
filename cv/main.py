import cv2
import sys
import numpy as np

emoji = cv2.imread("ws.png",cv2.IMREAD_UNCHANGED)
faceCascade = cv2.CascadeClassifier('haarcascades_cuda/haarcascade_frontalface_default.xml')

video_capture = cv2.VideoCapture(0)

emoji = cv2.resize(emoji, (100, 100), interpolation = cv2.INTER_AREA)
emoji_alpha_channel = emoji[:,:,3].astype(np.float32) / 255.0
emoji_rgb_channel = emoji[:,:,:3].astype(np.float32)

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
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # capture image every 5 seconds
        # send to your API
        # change emoji to display
    x=y=50
    height, width = emoji_alpha_channel.shape
    origin_img = frame[y:y+height, x:x+width].astype(np.float32)
    front_img = np.stack([cv2.multiply(emoji_rgb_channel[:,:,channel_idx], emoji_alpha_channel) for channel_idx in range(3)], axis=2)
    back_img = np.stack([cv2.multiply(origin_img[:,:,channel_idx], 1-emoji_alpha_channel) for channel_idx in range(3)], axis=2)
    frame[y:y+height, x:x+width] = (front_img + back_img).astype(np.int)
    # = emoji_alpha_channel * frame[y:y+height, x:x+width] + (1-emoji_alpha_channel) * emoji_rgb_channel

   # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()