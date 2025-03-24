import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter


def laplacian_of_gaussian(image):
    """Aplikácia Laplacian of Gaussian (LoG) filtru na obrázok."""

    # Definícia LoG jadra (5x5)
    log_kernel = np.array([
        [0, 0, -1, 0, 0],
        [0, -1, -2, -1, 0],
        [-1, -2, 16, -2, -1],
        [0, -1, -2, -1, 0],
        [0, 0, -1, 0, 0]
    ])

    # Konvolúcia s filtrom (bez použitia OpenCV)
    image_array = np.array(image.convert('L'), dtype=np.float32)  # Previesť na grayscale
    filtered_image = apply_convolution(image_array, log_kernel)

    return Image.fromarray(np.clip(filtered_image, 0, 255).astype(np.uint8))


def apply_convolution(image_array, kernel):
    """Aplikácia konvolučného jadra na obrázok."""
    h, w = image_array.shape
    kh, kw = kernel.shape
    pad = kh // 2

    # Pridanie paddingu
    padded_image = np.pad(image_array, pad, mode='constant', constant_values=0)
    output = np.zeros_like(image_array)

    # Aplikácia konvolúcie
    for i in range(h):
        for j in range(w):
            region = padded_image[i:i + kh, j:j + kw]
            output[i, j] = np.sum(region * kernel)

    return output


def plot_histogram(image):
    """Vykreslenie histogramu pre každý RGB kanál."""
    image_array = np.array(image)
    colors = ['red', 'green', 'blue']

    plt.figure(figsize=(12, 4))
    for i, color in enumerate(colors):
        plt.subplot(1, 3, i + 1)
        plt.hist(image_array[:, :, i].flatten(), bins=256, color=color, alpha=0.7)
        plt.title(f'{color.capitalize()} Histogram')
        plt.xlim(0, 255)

    plt.tight_layout()
    plt.show()


def main(image_path):
    """Načíta obrázok, aplikuje LoG a zobrazí výsledok s histogramami."""
    image = Image.open(image_path)
    log_image = laplacian_of_gaussian(image)

    # Zobrazenie výsledkov
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Pôvodný obrázok")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(log_image, cmap='gray')
    plt.title("LoG detekcia hrán")
    plt.axis("off")

    plt.show()

    plot_histogram(image)

main('Test_obrazok_1.jpg')
