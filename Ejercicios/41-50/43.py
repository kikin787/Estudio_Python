"""
Solicita al usuario ingresar un numero n
e imprime el factorial de ese numero
"""

n = int(input("Dame un numero: "))
fact = 1
while n > 0:
    fact *= n
    n -= 1
print(fact)