import cv2
import numpy as np

img = cv2.imread('coins.jpg')
if img is None:
    print("Chyba: Obrázok sa nepodarilo načítať.")


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


gray = cv2.GaussianBlur(gray, (9, 9), 2, 2)

#gray = cv2.medianBlur(gray, 5)


circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT_ALT, 1, 30, param1=100, param2=0.5, minRadius=10, maxRadius=150)

circles = np.uint16(np.around(circles))
for i in circles[0,:]:

    cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),2)

    cv2.circle(img,(i[0],i[1]),2,(0,0,255),3)

cv2.imshow("Circles", img)
cv2.waitKey(0)
cv2.destroyAllWindows()