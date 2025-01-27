'''
Obtener el cuadrado de la suma de dos lista de números con map
'''
def sum(a,b):
    return (a+b)**2
uno = [2,3,4,5]
dos = [6,7,8,9]

# suma = list(map(lambda n,b: (n+b)**2, uno, dos)) con lambda
suma = list(map(sum, uno, dos))
print(suma)