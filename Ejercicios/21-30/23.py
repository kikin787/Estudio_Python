"""
Verifica si una palabra es un palindromo
"""
# def es_palindromo(texto):
#     texto.lower().replace(" ", "")
#     texto1 = texto[::-1]
#     return texto == texto1
cadena = "Anita lava la tina"
cadenaCom = cadena.lower().replace(" ", "")
cadenaCom1 = cadenaCom[::-1]
print(f"La cadena {cadena} es: {cadenaCom == cadenaCom1}")

