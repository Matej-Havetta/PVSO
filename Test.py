
from ximea import xiapi
import cv2
import numpy as np





#mat = cv2.imread(cv2.samples.findFile("image0.jpg"))


#imgs = [mat, cv2.bitwise_not(mat), mat[:240, :320]]
#cv2.imshow("img", imgs)
#cv2.imwrite("test.jpg", imgs)



image1 = cv2.imread("image1.jpg")
image2 = cv2.imread("image2.jpg")
image3 = cv2.imread("image3.jpg")
image4 = cv2.imread("image4.jpg")



dimensions = image1.shape

height, width, num_channels = image2.shape

rot_image = image2.copy()
#for row in range(height):
#        for col in range(width):
#            rot_image[row][col], rot_image[col][row] = image2[col][row], image2[row][col]
rot_image = np.transpose(image2, (1, 0, 2))

red_image3 = image3.copy()
red_image3[:, :, 1:] = 0

top_row = np.concatenate((image1, rot_image), axis=1)
bottom_row = np.concatenate((red_image3, image4), axis=1)
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