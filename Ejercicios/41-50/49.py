"""
Simular el lanzamiento de un dado hasta lograr un 6
"""
import random
contador = 0
while True:
    dado = random.randint(1,6)
    if dado == 6:
        contador += 1
        print(f"Felicidades obtuviste un 6 en {contador} intentos, hasta pronto!!")
        break
    else:
        print(f"Lo siento obtuviste un {dado} sigue participando")
        contador += 1
