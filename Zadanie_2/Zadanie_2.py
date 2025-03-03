import cv2
import numpy as np
from ximea import xiapi

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
def calibrate_camera(image_files, pattern_size=(9, 6), square_size=0.025):
    obj_points = []
    img_points = []
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= square_size

    for fname in image_files:
        img = cv2.imread(fname)
        if img is None:
            print(f"Error: Unable to read image file {fname}")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        if ret:
            img_points.append(corners)
            obj_points.append(objp)
        else:
            print(f"Warning: Chessboard corners not found in image {fname}")

    if not img_points:
        raise ValueError("Error: No chessboard corners found in any image.")

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)
    if not ret:
        raise RuntimeError("Error: Camera calibration failed.")

    fx = mtx[0, 0]
    fy = mtx[1, 1]
    cx = mtx[0, 2]
    cy = mtx[1, 2]

    print(f'Camera Matrix:\n{mtx}')
    print(f'fx: {fx}, fy: {fy}, cx: {cx}, cy: {cy}')
    return mtx, dist


# Funkcia na detekciu kruhov pomocou Hough Transformácie
def detect_circles(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30, param1=50, param2=30, minRadius=10,
                               maxRadius=100)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv2.circle(image, (i[0], i[1]), 2, (0, 0, 255), 3)
            print(f'Kruh - Stred: ({i[0]}, {i[1]}), Priemer: {2 * i[2]} px')
    return image


# Hlavný kód
frame = capture_image()
images = ['img1.jpg','img2.jpg','img3.jpg','img4.jpg','img5.jpg','img6.jpg','img7.jpg','img8.jpg','img9.jpg','img10.jpg','img11.jpg','img12.jpg']
mtx, dist = calibrate_camera(images)
detected_circles = detect_circles(frame)
cv2.resizeWindow('Detected Circles', 800, 600)
cv2.imshow('Detected Circles', detected_circles)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Ukončenie kamery
cam.stop_acquisition()
cam.close_device()
