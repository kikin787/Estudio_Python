'''
filtrar numeros pares con filter 
'''
numeros = [1,2,3,4,5,6,7,8,9]

pares = list(filter(lambda n: n % 2 ==0, numeros))
print(pares)