import cv2

def image_to_matrix(image_path, output_txt_path):
    # Leer la imagen
    imagen = cv2.imread(image_path)

    # Obtener dimensiones de la imagen
    height, width, _ = imagen.shape

    # Inicializar una matriz para almacenar los valores
    matrix = [[0] * width for _ in range(height)]

    # Recorrer cada píxel de la imagen
    for y in range(height):
        for x in range(width):
            # Obtener el valor RGB del píxel
            b, g, r = imagen[y, x]

            # Calcular el valor promedio y almacenarlo en la matriz
            matrix[y][x] = int((r + g + b) / 3)

    # Escribir la matriz en un archivo de texto
    with open(output_txt_path, 'w') as txt_file:
        for row in matrix:
            txt_file.write('\t'.join(map(str, row)) + '\n')

if __name__ == "__main__":
    # Ruta de la imagen de entrada
    input_image_path = "GOAT.jpg"  # Reemplaza con la ruta de tu imagen
    
    # Ruta del archivo de salida .txt
    output_txt_path = "salida.txt"
    
    # Convertir la imagen a matriz y escribir en el archivo .txt
    image_to_matrix(input_image_path, output_txt_path)
