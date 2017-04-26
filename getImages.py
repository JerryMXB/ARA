from webcam import Webcam
import cv2
from datetime import datetime

webcam = Webcam()
webcam.start()

NROW = 7
NCOL = 6
while True:

    # get image from webcam
    image = webcam.get_current_frame()

    # display image
    cv2.imshow('grid', image)
    cv2.waitKey(1000)

    # save image to file, if pattern found
    ret, corners = cv2.findChessboardCorners(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (NROW, NCOL), None)

    if ret == True:
        print('get')
        filename = datetime.now().strftime('%Y%m%d_%Hh%Mm%Ss%f') + '.jpg'
        cv2.imwrite("posemac/sample_images/" + filename, image)
