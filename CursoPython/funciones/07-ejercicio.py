def es_palindromo(texto):
    texto = texto.lower().replace(" ", "")
    texto1 = texto[::-1]
    return texto == texto1

print("Abba", es_palindromo("Abba"))
print("Anita lava la tina", es_palindromo("Anita lava la tina"))
print("Macetas", es_palindromo("Macetas"))

# def no_space(texto):
#     resultado = ""
#     for char in texto:
#         if char != " ":
#             resultado += char
#     return resultado

# def reverse(texto):
#     texto_al_reves = ""
#     for char in texto:
#         texto_al_reves = char + texto_al_reves
#     return texto_al_reves

print("Anita lava la tina", es_palindromo("Anita lava la tina"))