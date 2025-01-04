"""
Realiza operaciones básicas con conjuntos unión e intersección
"""

conjuntoA = {1,2,3}
conjuntoB = {3,4,5}

union = conjuntoA | conjuntoB
interseccion = conjuntoA & conjuntoB
print(f"Unión del conjunto A: {conjuntoA} y conjunto B: {conjuntoB}: {union}")
print(f"Intersección del conjunto A: {conjuntoA} y conjunto B: {conjuntoB}: {interseccion}")