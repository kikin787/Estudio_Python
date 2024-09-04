import random
def cargas(sumaFinal, tamaño_ventana):
    numeros_random = []
    suma = 0
    while suma < sumaFinal:
        numeroA = random.randint(5, 15)
        if suma + numeroA > sumaFinal:
            numeros_random.append(sumaFinal - suma)
            break
        else:
            numeros_random.append(numeroA)
            suma = sum(numeros_random)
    if len(numeros_random) % tamaño_ventana != 0:
        resto = len(numeros_random) % tamaño_ventana
        if resto != 0:
            elementos_faltantes = tamaño_ventana - resto
            numeros_random.extend([0] * elementos_faltantes)
    return numeros_random

def generar_poblacion(num_individuos, tamaño_ventana):
    poblacion = []
    for _ in range(num_individuos):
        individuo = [random.uniform(1, tamaño_ventana) for _ in range(tamaño_ventana)]
        poblacion.append(individuo)
    return poblacion

def calcular_fitness(asignacion_procesos, maxspan):
    utilizacion_procesadores = []
    for procesos in asignacion_procesos.values():
        carga_total = sum(procesos)
        utilizacion = carga_total / maxspan
        utilizacion_procesadores.append(utilizacion)

    apu = sum(utilizacion_procesadores) / len(utilizacion_procesadores)
    fitness = (1 / maxspan) * apu
    return fitness

def seleccion_por_ruleta(poblacion, fitness_poblacion):
    total_fitness = sum(fitness_poblacion)
    probabilidad_seleccion = [fit / total_fitness for fit in fitness_poblacion]
    papa1 = random.choices(poblacion, weights=probabilidad_seleccion)[0]
    papa2 = random.choices(poblacion, weights=probabilidad_seleccion)[0]
    return papa1, papa2

def mutacion(individuo, tasa_mut):
    if random.random() < tasa_mut:
        idx1, idx2 = random.sample(range(len(individuo)), 2)
        individuo[idx1], individuo[idx2] = individuo[idx2], individuo[idx1]
    return individuo


def cruzamiento(papa1, papa2, tasa_cruz, tasa_mut):
    if random.random() < tasa_cruz:
        punto_de_cruce = random.randint(1, len(papa1) - 1)

        hijo1 = papa1[:punto_de_cruce] + [x for x in papa2 if x not in papa1[:punto_de_cruce]]
        hijo2 = papa2[:punto_de_cruce] + [x for x in papa1 if x not in papa2[:punto_de_cruce]]
    else:
        hijo1 = papa1
        hijo2 = papa2

    return mutacion(hijo1, tasa_mut), mutacion(hijo2, tasa_mut)


def conversion(individuo, carga, tamaño_ventana, num_procesadores):
    asignacion_procesos = {i + 1: [] for i in range(num_procesadores)}
    indices = list(map(int, individuo))  # No se usa split() porque `individuo` es una lista
    x = 0
    while x < len(carga):
        for i, indice in enumerate(indices):
            clave = (i % num_procesadores) + 1
            asignacion_procesos[clave].append(carga[(indice - 1) + x])
        x = x + tamaño_ventana
    maxspan = max(sum(procesos) for procesos in asignacion_procesos.values())
    fitness = calcular_fitness(asignacion_procesos, maxspan)
    return asignacion_procesos, fitness

def slide_window(procesos, size_w):
    ventanas = []
    for i in range(0, len(procesos), size_w):
        ventana = procesos[i:i + size_w]
        if len(ventana) < size_w:
            ventana += [0] * (size_w - len(ventana))
        ventanas.append(ventana)
    return ventanas

def calcular_estadisticas_procesadores(asignacion_procesos):
    colas_procesadores = [procesos for procesos in asignacion_procesos.values()]
    media_procesadores = sum(map(sum, colas_procesadores)) / len(colas_procesadores)
    max_cola = max(sum(procesos) for procesos in asignacion_procesos.values())
    return colas_procesadores, media_procesadores, max_cola

def imprimir_colas_procesadores(colas_procesadores):
    for i, cola in enumerate(colas_procesadores):
        print(f"Procesador {i} = {sum(cola)}")
def distribuir_con_slide_window():
    num_twits = 250000
    num_procesadores = 8
    tamaño_ventana = num_procesadores * 2
    num_individuos = 10
    num_generaciones = 50
    tasa_mut = 0.1
    tasa_cruz = 0.8

    while num_procesadores <= 8:
        carga = cargas(num_twits, tamaño_ventana)

        for generacion in range(num_generaciones):
            poblacion = generar_poblacion(num_individuos, tamaño_ventana)
            mejor_asignacion_procesos = None
            mejor_carga_actual = float('-inf')
            nueva = None

            for individuo in poblacion:
                asignacion, _ = conversion(individuo, carga, tamaño_ventana, num_procesadores)
                carga_procesadores = [sum(procesos) for procesos in asignacion.values()]
                max_carga_procesador = max(carga_procesadores)

                if max_carga_procesador > mejor_carga_actual:
                    mejor_carga_actual = max_carga_procesador
                    mejor_asignacion_procesos = asignacion.copy()
                    nueva = individuo

            print(f"\nGeneración: {generacion + 1}")
            print("Detalles de la asignación de procesos:")
            for i, procesos in mejor_asignacion_procesos.items():
                print(f"Procesador {i} = {sum(procesos)}")
            print(f"Media de las cargas de procesadores: {sum(carga_procesadores) / len(carga_procesadores)}")
            print(f"Mayor carga encontrada: {mejor_carga_actual}")
            print(f"Nueva población: {nueva}")

            nueva_poblacion = []

            for _ in range(num_individuos):
                if random.random() < tasa_cruz:
                    papa1, papa2 = seleccion_por_ruleta(poblacion, [1] * num_individuos)
                    hijo1, hijo2 = cruzamiento(papa1, papa2, tasa_cruz, tasa_mut)
                    nueva_poblacion.extend([hijo1, hijo2])

                nueva_poblacion.append(mutacion(nueva, tasa_mut))

            poblacion = nueva_poblacion

        num_procesadores += 1

distribuir_con_slide_window()