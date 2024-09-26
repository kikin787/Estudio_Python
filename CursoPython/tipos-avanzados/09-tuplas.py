numeros = (1, 2, 3) + (4, 5, 6)
print(numeros)

punto = list(numeros)

print(punto)

menosNumeros = numeros[:2]
print(menosNumeros)
primero, segundo, *otros = numeros
print(primero, segundo, otros)
primero = 10
print(primero, segundo, otros)
