n1 = int(input("Ingresa el primer numero: "));
n2 = int(input("Ingresa el segundo numero: "));

suma = n1 + n2
resta = n1 - n2
multi = n1 * n2
div = n1 / n2

mensaje = f"""
Para los números {n1} y {n2}.
El resultado de la suma es {suma}.
El resultado de la resta es {resta}.
El resultado de la multiplicación es {multi}.
El resultado de la división es {div}.
"""

print(mensaje)
