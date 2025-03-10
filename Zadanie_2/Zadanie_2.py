import cv2
import numpy as np
from ximea import xiapi
import glob

# Inicializácia kamery XIMEA
cam = xiapi.Camera()
cam.open_device()
cam.set_exposure(10000)  # Nastavenie expozície
cam.start_acquisition()


# Funkcia na získanie snímky z kamery XIMEA
def capture_image():
    img = xiapi.Image()
    cam.get_image(img)
    frame = img.get_image_data_numpy()
    return frame


# Funkcia na kalibráciu kamery pomocou šachovnice
def calibrate_camera(images, chessboardSize = (5, 7), square_size=0.025):
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1, 2)
    objp *= square_size

    # Arrays to store object points and image points from all the images.
    obj_points = []  # 3d point in real world space
    img_points = []  # 2d points in image plane.

    for image in images:
        img = cv2.imread(image)
        if img is None:
            print(f"Error: Unable to read image file {image}")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, chessboardSize, None)
        if ret:
            obj_points.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            img_points.append(corners2)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, chessboardSize, corners2, ret)
            cv2.imshow('img', img)
            cv2.waitKey(0)
        else:
            print(f"Warning: Chessboard corners not found in image {image}")

    if not img_points:
        raise ValueError("Error: No chessboard corners found in any image.")

    ret, cameraMatrix, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)
    print("cameraMatrix:", cameraMatrix)
    print("fx:", cameraMatrix.item(0, 0), "fy:", cameraMatrix.item(1, 1))
    print("cx:", cameraMatrix.item(0, 2), "cy:", cameraMatrix.item(1, 2))


# Funkcia na detekciu kruhov pomocou Hough Transformácie
def detect_circles(image, minDist=30, param1=50, param2=30, minRadius=10, maxRadius=100):
    image = cv2.medianBlur(image, 5)
    cimg = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, dp=1.2, minDist=minDist, param1=param1, param2=param2,
                               minRadius=minRadius, maxRadius=maxRadius)
    output = cimg.copy()
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv2.circle(output, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv2.circle(output, (i[0], i[1]), 2, (0, 0, 255), 3)
    return output

def nothing(x):
    pass

# Hlavný kód
images = glob.glob('*.jpg')
calibrate_camera(images)

cv2.namedWindow('Detected Circles')
cv2.createTrackbar('Min Distance', 'Detected Circles', 30, 200, nothing)
cv2.createTrackbar('Param1', 'Detected Circles', 50, 200, nothing)
cv2.createTrackbar('Param2', 'Detected Circles', 30, 200, nothing)
cv2.createTrackbar('Min Radius', 'Detected Circles', 10, 100, nothing)
cv2.createTrackbar('Max Radius', 'Detected Circles', 100, 200, nothing)

while True:
    cv2.imshow('Detected Circles', capture_image())
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    minDist = cv2.getTrackbarPos('Min Distance', 'Detected Circles')
    param1 = cv2.getTrackbarPos('Param1', 'Detected Circles')
    param2 = cv2.getTrackbarPos('Param2', 'Detected Circles')
    minRadius = cv2.getTrackbarPos('Min Radius', 'Detected Circles')
    maxRadius = cv2.getTrackbarPos('Max Radius', 'Detected Circles')
    frame = capture_image()
    if frame is not None:
        output = detect_circles(frame,minDist, param1, param2, minRadius, maxRadius)
        cv2.imshow('Detected Circles', output)
    else:
        print("Error: No image captured from camera.")

cv2.destroyAllWindows()

# Ukončenie kamery
cam.stop_acquisition()
cam.close_device()
