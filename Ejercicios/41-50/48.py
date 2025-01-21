"""
Simular el lanzamiento de una moneda
"""

import random

while True:
    moneda = random.randint(1,2)
    if moneda == 1:
        print("La moneda cayó cara")
    else:
        print("La moneda cayó cruz")
    jugar = (input("¿Deseas jugar otra vez? (S/N): "))
    if jugar.lower() == 's':
        print("Bienvenido de nuevo")
    else:
        print("Gracias por jugar")
        break