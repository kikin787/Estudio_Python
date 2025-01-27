'''
Filtrar elementos que son listas
'''

lista1 = ['Hola', [1,2,3], 'como', ['ee',2], 'estás']

lista = list(filter(lambda n: isinstance(n,list), lista1))
print(lista)
