"""
Verifica si una cadea es mayor 
o igual a 10 caracteres
"""

name = "Hola"

if len(name) > 10:
    print(f"La cadena {name} es mayor a 10 caracteres")
elif len(name) == 10:
    print(f"La cadena {name} es igual a 10 caracteres")
else:
    print(f"La cadenma {name} es menor a 10 caracteres")