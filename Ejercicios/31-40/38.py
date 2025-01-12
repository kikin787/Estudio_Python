"""
Determina si un numero es divisible entre 5 y 7
"""

n = int(input("Dame un número: "))

if n % 5 == 0 and n % 7 == 0:
    print(f"{n} es divisible entre 5 y 7")
else:
    print(f"{n} no es divisile entre 5 y 7")