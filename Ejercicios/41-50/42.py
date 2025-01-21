"""
solicita al usuario ingresar un numero N y luego
imprime la suma de todos los numeros desde 1 hasta N
"""

n = int(input("Dame el numero: "))

for e in range(n):
    n += e
    e += 1
print(n)