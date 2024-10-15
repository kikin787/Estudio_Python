# Leer la entrada
N, M = map(int, input().split())

# Parte superior del patron
for i in range(1, N, 2):  # Incrementos de 2 para generar patrones
    pattern = ('.|.' * i).center(M, '-')
    print(pattern)

# Parte central con la palabra "WELCOME"
print('WELCOME'.center(M, '-'))

# Parte inferior del patron (es el espejo de la parte superior)
for i in range(N-2, 0, -2):  # Decrementos de 2 para generar el patron invertido
    pattern = ('.|.' * i).center(M, '-')
    print(pattern)
