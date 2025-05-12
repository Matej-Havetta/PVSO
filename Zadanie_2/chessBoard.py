import numpy as np
import cv2 as cv
import glob

chessboardSize = (5, 7)
frameSize = (600, 600)

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('*.jpg')

for image in images:
    print(image)
    img = cv.imread(image)
    # cropped_image = img[75:475, 100:600]
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # plt.imshow(cropped_image, cmap="gray", vmin=0, vmax=255)
    # plt.show()
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 =  (gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        cv.drawChessboardCorners(img, chessboardSize, corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(0)

#cv.destroyAllWindows()

############ Calibration ######################

img = cv.imread('img9.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print("cameraMatrix:", cameraMatrix)
print("fx:", cameraMatrix.item(0, 0), "fy:", cameraMatrix.item(1, 1))
print("cx:", cameraMatrix.item(0, 2), "cy:", cameraMatrix.item(1, 2))
