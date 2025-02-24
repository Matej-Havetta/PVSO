from ximea import xiapi
import cv2
import numpy as np
### runn this command first echo 0|sudo tee /sys/module/usbcore/parameters/usbfs_memory_mb  ###

#create instance for first connected camera
cam = xiapi.Camera()



#start communication
#to open specific device, use:
#cam.open_device_by_SN('41305651')
#(open by serial number)
print('Opening first camerecho 0|sudo tee /sys/module/usbcore/parameters/usbfs_memory_mba...')
cam.open_device()

#settings
cam.set_exposure(100000)
cam.set_param('imgdataformat','XI_RGB32')
cam.set_param('auto_wb', 1)
print('Exposure was set to %i us' %cam.get_exposure())

#create instance of Image to store image data and metadata
img = xiapi.Image()

#start data acquisition
print('Starting data acquisition...')
cam.start_acquisition()

k = 0
while k != 4:
    cam.get_image(img)
    frame = img.get_image_data_numpy()
    frame = cv2.resize(frame, (240,240))
    cv2.imshow("camera window", frame)
    if cv2.waitKey(1) == ord(" "):
        cam.get_image(img)
        image = img.get_image_data_numpy()
        image = cv2.resize(image,(240,240))
        cv2.imshow("img{}".format(k+1), image)
        cv2.imwrite("img{}.jpg".format(k+1), image)
        k += 1
#stop data acquisition
print('Stopping acquisition...')
cam.stop_acquisition()

#stop communication
cam.close_device()

#read saved images
image1 = cv2.imread("img1.jpg")
image2 = cv2.imread("img2.jpg")
image3 = cv2.imread("img3.jpg")
image4 = cv2.imread("img4.jpg")


#get dimensions
dimensions = image1.shape
height, width, numchanel = dimensions

#copy image 2 for rotation
rot_image = image2.copy()
#rotate image 90 deg
for row in range(height):
        for col in range(width):
            rot_image[row][col], rot_image[col][row] = image2[col][row], image2[row][col]

# Split into channels
blue, green, red = cv2.split(image3)

# Create blank channel (black image of same shape)
blank = np.zeros_like(blue)

# Show only Red channel (set G & B to 0)
red_only = cv2.merge([blank, blank, red])

# Show only Green channel (set R & B to 0)
green_only = cv2.merge([blank, green, blank])

# Show only Blue channel (set R & G to 0)
blue_only = cv2.merge([blue, blank, blank])


top_row = np.concatenate((image1, rot_image), axis=1)
bottom_row = np.concatenate((red_only, image4), axis=1)
concatenated_image = np.concatenate((top_row, bottom_row), axis=0)

kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
concatenated_image[0:dimensions[0], 0:dimensions[1]] = cv2.filter2D(concatenated_image[0:dimensions[0], 0:dimensions[1]], -1, kernel)

cv2.imshow("img", concatenated_image)

cv2.imwrite("obrazok_final.jpg", concatenated_image)


h, w, n = concatenated_image.shape
print("datovy typ: " + str(concatenated_image.dtype))
print("rozmer: " + str(h) + "x" + str(w))
print("velkost: " + str(concatenated_image.size))

cv2.waitKey(0)

print('Done.')
