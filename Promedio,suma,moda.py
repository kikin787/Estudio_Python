import statistics

n = int(input('Valores que necesitas: '))
numeros = []

for i in range(n):
    valor = float(input(f"Ingrese el valor {i + 1}: "))
    numeros.append(valor)

sum = 0
promedio = 0
for prom in range(len(numeros)):
    sum += numeros[prom]

promedio = sum/n
moda = statistics.mode(numeros)

print(f"Promedio: {promedio} \n Suma: {sum} \n Moda: {moda}")
