"""
Escribe una funcion para calcular el promedio
de una lista de numeros
"""

def calcularPromedio(lista):
    return sum(lista)/len(lista)

numeros = [10,20,30,40,50]
print(calcularPromedio(numeros))