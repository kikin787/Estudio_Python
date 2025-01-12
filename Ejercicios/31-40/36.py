"""
Pide un caracter y define si es una vocal
"""

l = input("Dame una letra: ")
vocales = ['a', 'e', 'i', 'o', 'u'] 

if l.lower() in vocales:
    print(f"{l} es una vocal")
else:
    print(f"{l} no es una vocal")