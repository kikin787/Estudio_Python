"""
Solicita al usuario ingresar un numero y cuenta los digitos que tiene
"""

n = int(input("Dame un numero: "))
contar = len(str(n))

print(f"{n} tiene {contar} digitos")

contador = 0
a = n
while a != 0:
    a //= 10
    contador += 1
print(f"{n} tiene {contar} digitos") 