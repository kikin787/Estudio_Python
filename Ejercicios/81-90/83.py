'''
Calcular la longitud de una lista de palabras utilizando map
'''

def calcular(x):
    return len(x)

cade = ['hola', 'perro', 'gato', 'jirafa']
# longi = list(map(lambda n: len(n), cade)) con lambda
longi = list(map(calcular, cade))
print(longi)