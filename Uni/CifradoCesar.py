# Función para descifrar un texto cifrado con el Cifrado César
def descifrar_cesar(texto_cifrado, desplazamiento):
    resultado = ""
    for letra in texto_cifrado:
        # Verifica si es una letra
        if letra.isalpha():
            # Convierte la letra a minúscula para manejar tanto mayúsculas como minúsculas
            letra_minuscula = letra.lower()
            # Calcula la posición original de la letra
            indice_original = ord(letra_minuscula) - ord('a')
            # Aplica el desplazamiento inverso (descifrado)
            nuevo_indice = (indice_original - desplazamiento) % 26
            # Convierte el nuevo índice en letra
            nueva_letra = chr(nuevo_indice + ord('a'))
            # Conserva el formato original (mayúscula o minúscula)
            if letra.isupper():
                resultado += nueva_letra.upper()
            else:
                resultado += nueva_letra
        else:
            # Si no es letra (espacio, signo de puntuación), lo mantiene igual
            resultado += letra
    return resultado

# Ejemplo de uso
texto_cifrado = "Wklv lv d qxhyh dswdfr"  # Texto cifrado
desplazamiento = 3  # Desplazamiento utilizado en el cifrado

texto_descifrado = descifrar_cesar(texto_cifrado, desplazamiento)
print("Texto descifrado:", texto_descifrado)