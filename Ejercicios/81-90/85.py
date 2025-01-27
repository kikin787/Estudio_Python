'''
contar el numero de vocales en una lista de palabras utilizando map
'''

def vocales(a):
    # n = 0
    # for e in a:
    #     if e.lower() in 'aeiou':
    #         n += 1
    return sum(1 for e in a if e.lower() in 'aeiou')

lista = ['perro', 'gato', 'naco', 'soloe']
res = list(map(vocales, lista))
# vocales = lambda palabra: sum(1 for letra in palabra if letra in 'aeiou') con lambda

print(res)