import cv2
import numpy as np
import matplotlib.pyplot as plt

# Načítanie obrázka
img = cv2.imread('Test_obrazok_1.jpg')

# Konverzia do odtieňov sivej a následne do typu int16
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.int16)

# Definícia Laplacovho jadra (mimo výpočtu, len pre referenciu)
laplacian_kernel = np.array([[0, 1, 0],
                             [1, -4, 1],
                             [0, 1, 0]])

# Horizontálne a vertikálne jadrá (rovnaké v tomto prípade)
horizontal = np.array([[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]])

vertical = np.array([[0, 1, 0],
                     [1, -4, 1],
                     [0, 1, 0]])

# Rozmery obrázka
h, w = gray_img.shape

# Inicializácia výstupných obrazov
newhorizontalImage = np.zeros((h, w), dtype=np.float32)
newverticalImage = np.zeros((h, w), dtype=np.float32)
newgradientImage = np.zeros((h, w), dtype=np.float32)

# Manuálne násobenie a konvolúcia
for i in range(1, h - 1):
    for j in range(1, w - 1):
        horizontalGrad = (horizontal[0, 0] * gray_img[i - 1, j - 1]) + \
                         (horizontal[0, 1] * gray_img[i - 1, j]) + \
                         (horizontal[0, 2] * gray_img[i - 1, j + 1]) + \
                         (horizontal[1, 0] * gray_img[i, j - 1]) + \
                         (horizontal[1, 1] * gray_img[i, j]) + \
                         (horizontal[1, 2] * gray_img[i, j + 1]) + \
                         (horizontal[2, 0] * gray_img[i + 1, j - 1]) + \
                         (horizontal[2, 1] * gray_img[i + 1, j]) + \
                         (horizontal[2, 2] * gray_img[i + 1, j + 1])

        verticalGrad = (vertical[0, 0] * gray_img[i - 1, j - 1]) + \
                       (vertical[0, 1] * gray_img[i - 1, j]) + \
                       (vertical[0, 2] * gray_img[i - 1, j + 1]) + \
                       (vertical[1, 0] * gray_img[i, j - 1]) + \
                       (vertical[1, 1] * gray_img[i, j]) + \
                       (vertical[1, 2] * gray_img[i, j + 1]) + \
                       (vertical[2, 0] * gray_img[i + 1, j - 1]) + \
                       (vertical[2, 1] * gray_img[i + 1, j]) + \
                       (vertical[2, 2] * gray_img[i + 1, j + 1])

        # Manuálne aplikovaný Laplacov operátor
        laplace = (-1 * gray_img[i + 1, j]) + \
                  (-1 * gray_img[i - 1, j]) + \
                  (-1 * gray_img[i, j + 1]) + \
                  (-1 * gray_img[i, j - 1]) + \
                  (-1 * gray_img[i - 1, j - 1]) + \
                  (-1 * gray_img[i + 1, j + 1]) + \
                  (-1 * gray_img[i - 1, j + 1]) + \
                  (-1 * gray_img[i + 1, j - 1]) + \
                  (8 * gray_img[i, j])

        newhorizontalImage[i, j] = abs(horizontalGrad)
        newverticalImage[i, j] = abs(verticalGrad)
        newgradientImage[i, j] = laplace

# Orezanie a konverzia na uint8
newgradientImage_clipped = np.clip(newgradientImage, 0, 255).astype(np.uint8)

# Zobrazenie a uloženie výsledku
plt.figure()
plt.title('Detekcia hrán - Laplace')
plt.imsave('result.png', newgradientImage_clipped, cmap='gray', format='png')
plt.imshow(newgradientImage_clipped, cmap='gray')
plt.show()

# Zobrazenie pomocou OpenCV
cv2.imshow('Laplacian', newgradientImage_clipped)
cv2.waitKey(0)
cv2.destroyAllWindows()