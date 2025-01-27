'''
filtrar cadenas de longitud mayor que 3 de una lista, usando filter
'''

cadenas = ['hola', 'yes', 'soy', 'kikin', 'perro']

mayor = list(filter(lambda n: len(n) > 3, cadenas))
print(mayor)