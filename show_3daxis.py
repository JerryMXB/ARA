from webcam import Webcam
import numpy as np
import cv2
import sys
from PIL import Image

print(cv2.__version__)

NROW = 6
NCOL = 4
QUADRILATERAL_POINTS = 4


def order_points(points):
    s = points.sum(axis=1)
    diff = np.diff(points, axis=1)

    ordered_points = np.zeros((4, 2), dtype="float32")

    ordered_points[0] = points[np.argmin(s)]
    ordered_points[2] = points[np.argmax(s)]
    ordered_points[1] = points[np.argmin(diff)]
    ordered_points[3] = points[np.argmax(diff)]

    return ordered_points


def max_width_height(points):
    (tl, tr, br, bl) = points

    top_width = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    bottom_width = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    max_width = max(int(top_width), int(bottom_width))

    left_height = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    right_height = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    max_height = max(int(left_height), int(right_height))

    return (max_width, max_height)


def topdown_points(max_width, max_height):
    return np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]], dtype="float32")


def get_topdown_quad(image, src):
    # src and dst points
    src = order_points(src)

    (max_width, max_height) = max_width_height(src)
    dst = topdown_points(max_width, max_height)

    # warp perspective
    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, matrix, max_width_height(src))

    # return top-down quad
    return warped


def add_substitute_quad(image, substitute_quad, dst):
    # dst (zero-set) and src points
    dst = order_points(dst)

    (tl, tr, br, bl) = dst
    min_x = min(int(tl[0]), int(bl[0]))
    min_y = min(int(tl[1]), int(tr[1]))

    for point in dst:
        point[0] = point[0] - min_x
        point[1] = point[1] - min_y

    (max_width, max_height) = max_width_height(dst)
    src = topdown_points(max_width, max_height)

    # warp perspective (with white border)
    substitute_quad = cv2.resize(substitute_quad, (max_width, max_height))

    warped = np.zeros((max_height, max_width, 3), np.uint8)
    warped[:, :, :] = 255

    matrix = cv2.getPerspectiveTransform(src, dst)
    cv2.warpPerspective(substitute_quad, matrix, (max_width, max_height), warped, borderMode=cv2.BORDER_TRANSPARENT)

    # add substitute quad
    image[min_y:min_y + max_height, min_x:min_x + max_width] = warped

    return image


def resize_image(image, new_size):
    ratio = new_size / image.shape[1]
    return cv2.resize(image, (int(new_size), int(image.shape[0] * ratio)))


def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w / 2, h / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, rotation_matrix, (w, h))


def overlayimg(img, imgpts):
    imgpts = np.int32(imgpts).reshape(-1, 2)
    #substitute_image = cv2.imread('tuanzi.png')
    cap = cv2.VideoCapture('ad.mp4')
    ret, frame = cap.read()
    print(imgpts[0:4])
    #image = add_substitute_quad(img, substitute_image, imgpts[0:4])
    image = add_substitute_quad(img, frame, imgpts[0:4])
    return image


def overlayv(img, imgpts,video):
    imgpts = np.int32(imgpts).reshape(-1, 2)
    #substitute_image = cv2.imread('tuanzi.png')
    ret, frame = video.read()
    print(imgpts[0:4])
    #image = add_substitute_quad(img, substitute_image, imgpts[0:4])
    image = add_substitute_quad(img, frame, imgpts[0:4])
    return image


def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())

    cv2.line(img, corner, tuple(imgpts[0].ravel()), (255, 0, 0), 5)
    cv2.line(img, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 5)
    cv2.line(img, corner, tuple(imgpts[2].ravel()), (0, 0, 255), 5)
    return img


def drawcube(img, imgpts):
    imgpts = np.int32(imgpts).reshape(-1, 2)
    # draw ground floor in green
    cv2.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), -3)
    print(imgpts[:4]);
    # draw pillars in blue
    for i, j in zip(range(4), range(4, 8)):
        cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255), 3)
    # draw top layer in red
    cv2.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 3)
    return img

def on_bar(x):

    pass

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((NROW * NCOL, 3), np.float32)
objp[:, :2] = np.mgrid[0:NROW, 0:NCOL].T.reshape(-1, 2)
axis = np.float32([[0, 0, 0], [0, 3, 0], [3, 3, 0], [3, 0, 0],  #cube
[0, 0, -3], [0, 3, -3], [3, 3, -3], [3, 0, -3]])

with np.load('posemac/webcam_calibration_ouput.npz') as X:
    mtx, dist, _, _ = [X[i] for i in ('mtx', 'dist', 'rvecs', 'tvecs')]

webcam = Webcam()
webcam.start()
cap1 = cv2.VideoCapture('coke.mp4')
cap2 = cv2.VideoCapture('ad.mp4')
cv2.namedWindow('img')

# init bar window
cv2.createTrackbar('Age','img',0,100,on_bar);

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
        #img = drawcube(img, imgpts)
        age = cv2.getTrackbarPos('Age','img')
        if age > 18:
            image = overlayv(img, imgpts, cap1)
        else:
            image = overlayv(img, imgpts, cap2)
        #image = overlayimg(img,imgpts)
        if image is not None:
            cv2.imshow('img', image)
            cv2.waitKey(10)
        else:
            print('but I cannot display!')
    else:
        cv2.imshow('img', img)
        cv2.waitKey(10)

