'''
convertir una lista de cadenas que sean numeros a enteros utilizando map
'''
def convertir_entero(x):
    return int(x)

numCa = ['1', '2', '3', '4']
strToInt = list(map(convertir_entero, numCa))
# strToInt = list(map(lambda n: int(n), numCa)) con lambda
print(numCa, strToInt)