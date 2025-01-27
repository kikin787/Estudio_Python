'''
Filtrar cadenas que contienen un caracter especifico usando filter
'''

lista = ['hola', 'perro', 'perra', 'gata', 'kinkon']

o = list(filter(lambda n: 'o'in n, lista))
print(o)