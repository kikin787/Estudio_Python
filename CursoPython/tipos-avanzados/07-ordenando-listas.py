numeros = [1, 9, 45, 3, 2, 6, 8]
mascotas = [
    "Pelusa", 
    "Wolfgang",
    "Pulga", 
    "Felipe", 
    "Wolfgang",
    "Kikin"
]

# mascotas.sort()
# numeros.sort()
# numeros.sort(reverse=True)

numeros2 = sorted(numeros, reverse=True)
print(numeros)
print(numeros2)

usuarios = [
    ["Chanchito", 4], 
    ["Felipe", 1], 
    ["Pulga", 5]
]

usuarios.sort(key=lambda el: el[2], reverse=True)
print(usuarios)