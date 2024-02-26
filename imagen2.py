import cv2

#abrir imagen
imagen = cv2.imread("GOAT.jpg")

#numero de pixles
print(f"Número de pixeles: {imagen.size}")

#conversion BGR a RGB
imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

#imprimir imagen
print(imagen_rgb)

#dimensiones de la imagen
alto, ancho, canales = imagen_rgb.shape
print(f"Tamaño de la imagen: {alto}, {ancho}")
print(f"Número de canales: {canales}")

#valor RGB de un pixel en una posición determinada
x = 20
y = 3
r, g, b = imagen_rgb[x,y]
print(f"Valores RGB en {x,y}: ")

#valor RGB de los primeros 10x10 pixeles
for i in range(10):
    for j in range(10):
        pixel = imagen_rgb[i,j]
        print(f"Pixel({i}, {j}): {pixel}")
    print("")

cv2.imshow('Imagen Origianl', imagen)