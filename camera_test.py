from ximea import xiapi
import cv2
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
        cv2.imwrite("images/img{}.jpg".format(k+1), image)
        k += 1
#stop data acquisition
print('Stopping acquisition...')
cam.stop_acquisition()

#stop communication
cam.close_device()

print('Done.')
