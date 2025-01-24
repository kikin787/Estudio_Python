"""
Funcion para verificar si un numero es par o impar
"""

def parImpar(a):
    if a % 2 == 0:
        return f"{a} es par"
    else:
        return f"{a} es impar"
print(parImpar(2), parImpar(3))