import cv2

#abrir imagen
imagen = cv2.imread("GOAT.jpg")

#numero de pixeles
print(f"Numero de pixeles: {imagen.size}")