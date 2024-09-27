from pprint import pprint
palabra = "abababc"

# def no_espacios(palabra):
#     list = []
#     for char in palabra:
#         if char != " ":
#             list += char
#     return list

def no_espacios(texto):
    return [char for char in texto if char != " "]

def contar_caracteres(cadena):
    contador = {}
    
    for char in cadena:
        if char in contador:
            contador[char] += 1
        else:
            contador[char] = 1
    return contador

def diccionario_a_tuplas(diccionario):
    return sorted(
        diccionario.items(),
        key=lambda key: key[1],
        reverse=True)

def encontrar_mayores(tuplas):
    mayor = tuplas[0][1]
    mayores = {}
    for tupla in tuplas:
        if mayor > tupla[1]:
            break
        mayores[tupla[0]] = tupla[1]
    return mayores

def mensaje(mayores):
    mensaje = "Los que más se repiten son: "
    for llave, valor in mayores.items():
        mensaje += f"\n- {llave} con {valor} repeticiones."
    print(mensaje)

resultado = no_espacios(palabra)
print(resultado)
resultado = contar_caracteres(resultado)
pprint(resultado, width=1)
tuplas = diccionario_a_tuplas(resultado)
pprint(tuplas)
mayorTuplas = encontrar_mayores(tuplas)
print(mayorTuplas)
message = mensaje(mayorTuplas)


    



