from webcam import Webcam
import numpy as np
import cv2

print(cv2.__version__)

NROW = 6
NCOL = 4


def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())

    cv2.line(img, corner, tuple(imgpts[0].ravel()), (255, 0, 0), 5)
    cv2.line(img, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 5)
    cv2.line(img, corner, tuple(imgpts[2].ravel()), (0, 0, 255), 5)
    return img


def drawcube(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1, 2)
    # draw ground floor in green
    cv2.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), -3)
    # draw pillars in blue
    for i, j in zip(range(4), range(4, 8)):
        cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255), 3)
    # draw top layer in red
    cv2.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 3)
    return img

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((NROW * NCOL, 3), np.float32)
objp[:, :2] = np.mgrid[0:NROW, 0:NCOL].T.reshape(-1, 2)
axis = np.float32([[0, 0, 0], [0, 3, 0], [3, 3, 0], [3, 0, 0],  #cube
[0, 0, -3], [0, 3, -3], [3, 3, -3], [3, 0, -3]])

with np.load('pose/webcam_calibration_ouput.npz') as X:
    mtx, dist, _, _ = [X[i] for i in ('mtx', 'dist', 'rvecs', 'tvecs')]

webcam = Webcam()
webcam.start()

while True:
    # get image from webcam
    img = webcam.get_current_frame()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # adapted to inner corners
    ret, corners = cv2.findChessboardCorners(gray, (NROW, NCOL), None)

    if ret == True:
        print('I find you')
        cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        # Find the rotation and translation vectors.
        try:
            rvecs, tvecs, _ = cv2.solvePnPRansac(objp, corners, mtx, dist)
        except:
            _, rvecs, tvecs = cv2.solvePnP(objp, corners, mtx, dist)

        # project 3D points to image plane
        imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
        img = drawcube(img, corners, imgpts)
        if img is not None:
            cv2.imshow('img', img)
            cv2.waitKey(10)
        else:
            print('but I cannot display!')
    else:
        cv2.imshow('img', img)
        cv2.waitKey(10)

