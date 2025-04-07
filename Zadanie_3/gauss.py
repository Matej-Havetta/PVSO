import cv2
import numpy as np
import matplotlib.pyplot as plt

# Načítanie a konverzia obrázka
img = cv2.imread('img1.jpg')
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.int16)

h, w = gray_img.shape

# Gaussian filter
gaussian_filter = np.array([[0.003765, 0.015019, 0.023792, 0.015019, 0.003765],
                            [0.015019, 0.059912, 0.094907, 0.059912, 0.015019],
                            [0.023792, 0.094907, 0.150342, 0.094907, 0.023792],
                            [0.015019, 0.059912, 0.094907, 0.059912, 0.015019],
                            [0.003765, 0.015019, 0.023792, 0.015019, 0.003765]])

# Padding
padded_image = np.pad(gray_img, ((2, 2), (2, 2)), mode='constant', constant_values=0)

# Aplikácia Gaussovho filtra
blur_image = np.zeros((h, w), dtype=np.int16)
for i in range(2, h+2):
    for j in range(2, w+2):
        region = padded_image[i-2:i+3, j-2:j+3]
        blur_image[i-2][j-2] = int(np.sum(region * gaussian_filter))

# Výstupné polia
laplacian = np.zeros((h, w), dtype=np.int16)
laplacian_diagonal = np.zeros((h, w), dtype=np.int16)

# Aplikácia Laplacovho operátora (manuálne)
for i in range(1, h-1):
    for j in range(1, w-1):
        laplacian_sum = (-1 * blur_image[i + 1][j]) + \
                        (-1 * blur_image[i - 1][j]) + \
                        (-1 * blur_image[i][j + 1]) + \
                        (-1 * blur_image[i][j - 1]) + \
                        (4 * blur_image[i][j])

        laplacian_diagonal_sum = (-1 * blur_image[i + 1][j]) + \
                                 (-1 * blur_image[i - 1][j]) + \
                                 (-1 * blur_image[i][j + 1]) + \
                                 (-1 * blur_image[i][j - 1]) + \
                                 (-1 * blur_image[i - 1][j - 1]) + \
                                 (-1 * blur_image[i + 1][j + 1]) + \
                                 (-1 * blur_image[i - 1][j + 1]) + \
                                 (-1 * blur_image[i + 1][j - 1]) + \
                                 (8 * blur_image[i][j])

        laplacian[i][j] = laplacian_sum
        laplacian_diagonal[i][j] = laplacian_diagonal_sum

# Orezanie na rozsah 0–255
laplacian_img = np.clip(laplacian, 0, 255).astype(np.uint8)
laplacian_diag_img = np.clip(laplacian_diagonal, 0, 255).astype(np.uint8)

# OpenCV verzia Laplace filtra (referenčná)
blurred = cv2.GaussianBlur(img, (5, 5), 0)
gray_blurred = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
laplacian_cv = cv2.Laplacian(gray_blurred, cv2.CV_64F, ksize=3)
laplacian_cv = np.uint8(np.absolute(laplacian_cv))

# Zobrazenie
cv2.imshow('laplacian_cv (OpenCV)', laplacian_cv)
cv2.imshow('Laplacian (manuálny)', laplacian_img)
cv2.imshow('Laplacian s uhlopriečkami (manuálny)', laplacian_diag_img)
plt.imsave('laplacian_cv (OpenCV)', laplacian_cv, cmap='gray', format='png')
plt.imsave('Laplacian (manuálny)', laplacian_img, format='png')
plt.imsave('Laplacian s uhlopriečkami (manuálny)', laplacian_diag_img, cmap='gray', format='png')
cv2.waitKey(0)
cv2.destroyAllWindows()