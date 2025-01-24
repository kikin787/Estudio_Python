"""
Escribe una funcion para calcular la tasa
de desempleo
td = no_desempleados/fuerza_laboralx100
"""

def td():
    noD = int(input("Dame los no desempleados: "))
    fue = float(input("Dame la fuerza laboral: "))
    return noD / fue * 100

print(f"La tasa de desempleo es: {td()}")