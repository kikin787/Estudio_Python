"""
Imprime la tabla de multiplicar de un numero ingresado por el usuario
"""

n = int(input("Dame un numero: "))

i = 1
while i <= 10:
    print(f"{i}x{n}={n * i}")
    i += 1