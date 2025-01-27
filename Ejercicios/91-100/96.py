'''
Crear una excepción que me ayude a determinar si el indice de una lista esta fuera de rango
'''

lista = [1,2,3]

try:
    print(lista[4])
except IndexError:
    print("El indice no existe")