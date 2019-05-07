import imutils
import cv2
import os

# face detection model
detector = cv2.CascadeClassifier(os.path.join('models','haarcascade_frontalface_default.xml'))


def detect_faces(frame):
    ''' Detects faces from captured image frames. '''
    frame = imutils.resize(frame, width=400) #decreasing the resolution for performance
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector.detectMultiScale(gray, scaleFactor=1.1,minNeighbors=5, minSize=(20, 20))
    b_boxes = [(x, y, x + w, y + h) for (x, y, w, h) in rects]
    face_crops=[frame[top:bottom, left:right] for (left, top, right, bottom) in b_boxes]
    # self.fps.update()

    return face_crops
