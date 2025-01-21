"""
Genera un numero aleatorio entre 1 y 10 
Luego, pide al usuario adivinar el numero hasta que lo haga correctamente
"""
import random

n = random.randint(1,10)
nU = int(input("Dame un numero: "))
intentos = 1
while nU != n:
    print(f"{nU} es diferente a {n}, sigue intentado")
    intentos += 1
    nU = int(input("Dame otro numero: "))

print(f"Felicidades {nU} es igual a {n}. \n Lo lograste en {intentos} intentos")

# while True:
#     nU = int(input("Dame un numero: "))
#     intentos += 1
#     if nU == n:
#         print(f"Felicidades {nU} es igual a {n}. \n Lo lograste en {intentos} intentos")
#         break
    