"""
Determina si un año es bisiesto
reglas de bisiesto
    - Divisible por 4.
    - No divisble por 100
    - Disible por 400
"""

anio = int(input("Dame el año que quieres comprobar: "))

if (anio % 4 == 0 and anio % 100 != 0) or (anio % 400 == 0):
    print(f"{anio} es un año bisiesto")
else:
    print(f"{anio} no es un año bisiesto")