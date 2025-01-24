"""
Imprimir la suma de los numeros pares del 1 al 10
utilizando el ciclo for
"""

res = 0
for i in range(1,11):
    if i % 2 == 0:
        res += i
print(res)