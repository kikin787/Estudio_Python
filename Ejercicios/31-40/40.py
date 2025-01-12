"""
Calcula el imc de una persona
"""

peso = float(input("Dame tu peso en kg: "))
altura = float(input("Dame tu altura: "))

imc = peso / pow(altura, 2)

if imc < 18.5:
    print("Bajo peso")
elif 24.9 >= imc >= 18.5:
    print("Peso normal")
elif 29.9 >= imc >= 25:
    print("Sobrepeso")
elif 34.9 >= imc >= 30:
    print("Obesidad grado I")
elif 39.9 >= imc >= 35:
    print("Obesidad grado II")
elif imc >= 40:
    print("Obesidad grado III")