"""
Pedi un numero y verifica si es positivo, negativo o cero
"""

n = float(input("Dame un numero: "))

if n > 0:
    print(f"{n} es positivo")
elif n < 0:
    print(f"{n} es negativo")
else:
    print(f"{n} es 0")