"""
Verifica si la palbra ingresada es python
"""

palabra = "python"
character = input("Dame la palabra: ")

if character.lower() == palabra:
    print(f"{character.lower()} es igual a {palabra}")
else:
    print(f"{character.lower()} es diferente a {palabra}")