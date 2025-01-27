'''
elevar al cuadrado una lista de numeros utilizando map()
'''
def cuadrado(x):
    return x ** 2
numeros = [1,2,3,4,5]
# resultado = list(map(lambda n: n ** 2, numeros)) con función lambda

resultado = list(map(cuadrado, numeros))

print(resultado)