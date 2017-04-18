import numpy as np
import cv2
import glob

NROW = 8
NCOL = 6
# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
# eg: checkerboard: inner points
objp = np.zeros((NCOL * NROW, 3), np.float32)
objp[:, :2] = np.mgrid[0:NROW, 0:NCOL].T.reshape(-1, 2)


# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.
print('start')
# images = glob.glob('pose\\sample_images\\*.jpg')
images = glob.glob('posemac/sample_images/*.jpg')
for fname in images:
    print(fname)
    img = cv2.imread(fname)
    if img is None:
        print('It is empty')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (NROW, NCOL), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, (NROW, NCOL), corners, ret)
        cv2.imshow('image', img)
        cv2.waitKey(5)

# calibrate webcam and save output
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
np.savez("posemac/webcam_calibration_ouput", ret=ret, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)

cv2.destroyAllWindows()
