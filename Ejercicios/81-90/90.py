'''
duplicar cada elemento de una lista usando map y lambda
'''
lista = [2,3,4,5,6]
doble = list(map(lambda n: n * 2, lista))
print(doble)