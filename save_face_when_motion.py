import numpy as np
import cv2

from detector_haar import detect_faces
from IO import IO

cap = cv2.VideoCapture(0)

io = IO('cropped_faces')

fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

while(1):
    ret, frame = cap.read()

    fgmask = fgbg.apply(frame)
    test = np.sum(fgmask.flatten())
    print(test)
    if test > 600000:
        print("motion")
        images = detect_faces(frame)
        io.save_image(images)
    else:
        print("no")
    cv2.imshow('frame', fgmask)
    
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()