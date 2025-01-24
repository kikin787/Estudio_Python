"""
Pedir al usuario un numero e imprimir la tabla de multiplicar
del mismo
"""

n = int(input("Dame un numero para la tabla de multiplicar: "))

for i in range(1,11):
    print(f"{i}x{n}={i*n}")