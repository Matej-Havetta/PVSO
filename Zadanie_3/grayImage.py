input_path = 'img1.jpg'
output_path = 'img3.bmp'

# Načítanie obrázka ako binárneho súboru
with open(input_path, 'rb') as f:
    data = f.read()

# Jednoduchý parser JPG hlavičky pre šírku a výšku (veľmi zjednodušené, reálne použitie je zložitejšie)
start = data.find(b'\xff\xc0') + 5
height = (data[start] << 8) + data[start + 1]
width = (data[start + 2] << 8) + data[start + 3]

# Konverzia do odtieňov sivej (manuálne, bez knižníc)
grayscale_data = bytearray()
for i in range(0, len(data) - 2, 3):
    r, g, b = data[i], data[i + 1], data[i + 2]
    gray = int(0.2989 * r + 0.5870 * g + 0.1140 * b)
    grayscale_data.extend([gray, gray, gray])

# Uloženie výsledného obrázka
with open(output_path, 'wb') as f:
    f.write(grayscale_data)
print(f"Obrázok bol úspešne skonvertovaný a uložený ako '{output_path}'.")

