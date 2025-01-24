"""
Escribe una funcion para clasificar si una sustancia es acida, basica o neutra
apartir de su pH
"""

def sustancia(p):
    if p < 7:
        return "ácida"
    elif p > 7 :
        return "básica"
    else:
        return "neutra"

print(sustancia(8), sustancia(7), sustancia(6))