"""
Calcula el area de un rectangulo pide base y altura al usuario
"""

base = float(input("Dame la base del rectángulo: "))
altura = float(input("Dame la altura del rectángulo: "))
resultado = base * altura
print(f"El área del rectángulo con base de {base} y altura de {altura} es de: {round(resultado, 2)}")