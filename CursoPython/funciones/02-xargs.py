def suma(*numeros):
    resultado = 0
    for numero in numeros:
        resultado += numero
    print(resultado)


suma(2,7)
suma(2, 7, 9)
suma(2, 7, 15, 36, 52)