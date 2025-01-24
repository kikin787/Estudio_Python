"""
Escribe una función para calcular el volumen de un cilindro
"""

import math

def volumen(r,h):
    return math.pi*(r**2)*h

print(round(volumen(3,5),2))