"""
Escribe una función que pida la distancia y la velocidad para poder calcular el tiempo de viaje
"""

def tiempo(v,d):
    return f"El tiempo que tardó con vecolidad de {v} km/h con distancia de {d}m es: {d / v}s"

v = float(input("Dame la velocidad: "))
d = float(input("Dame la distancia: "))

print(tiempo(v,d))