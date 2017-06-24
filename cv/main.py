import cv2
import sys

s_img = cv2.imread("ws.png")
faceCascade = cv2.CascadeClassifier('haarcascades_cuda/haarcascade_frontalface_default.xml')

video_capture = cv2.VideoCapture(0)

s_img = cv2.resize(s_img, (100, 100), interpolation = cv2.INTER_AREA)
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
    frame[y:y+s_img.shape[0], x:x+s_img.shape[1]] = s_img

   # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()