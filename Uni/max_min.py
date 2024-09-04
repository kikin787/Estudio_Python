import random

# Parámetros del problema
num_tareas = 10
num_recursos = 5
num_hormigas = 10
max_tiempo_tarea = 10
max_costo_tarea = 100
max_feromona = 100

# Función para inicializar las feromonas
def inicializar_feromonas():
    return [[[random.uniform(0, max_feromona) for _ in range(num_tareas)] for _ in range(num_recursos)] for _ in range(num_hormigas)]

# Función para seleccionar una tarea basada en las feromonas y otras heurísticas
def seleccionar_tarea(hormiga, disponibles, feromonas, alfa, beta, maximizar=True):
    probabilidades = [sum(feromonas[hormiga][recurso][tarea] ** alfa * ((1 / feromonas[hormiga][recurso][tarea]) ** beta) for recurso in range(num_recursos)) for tarea in disponibles]
    if maximizar:
        probabilidades = [1 / p for p in probabilidades]  # Invertir probabilidades para maximización
    suma_probabilidades = sum(probabilidades)
    probabilidades = [p / suma_probabilidades for p in probabilidades]
    tarea_seleccionada = random.choices(disponibles, probabilidades)[0]
    return tarea_seleccionada

# Función para actualizar las feromonas
def actualizar_feromonas(solucion_hormiga, feromonas, maximizar=True):
    if maximizar:
        delta = 1
    else:
        delta = -1
    for tarea in solucion_hormiga:
        for i in range(num_hormigas):
            for recurso in range(num_recursos):
                feromonas[i][recurso][tarea] = max(0, min(max_feromona, feromonas[i][recurso][tarea] + delta))  # Ajustar las feromonas dentro del rango permitido
                # Normalizar las feromonas para mantenerlas dentro del rango permitido
                suma_feromonas = sum(feromonas[i][recurso])
                feromonas[i][recurso] = [feromona * max_feromona / suma_feromonas for feromona in feromonas[i][recurso]]

# Algoritmo de colonia de hormigas
def colonia_de_hormigas(num_iteraciones, alfa, beta, maximizar=True):
    feromonas = inicializar_feromonas()  # Generar feromonas iniciales
    for _ in range(num_iteraciones):
        for hormiga in range(num_hormigas):
            solucion_hormiga = []
            recursos_disponibles = list(range(num_tareas))
            while recursos_disponibles:
                tarea_seleccionada = seleccionar_tarea(hormiga, recursos_disponibles, feromonas, alfa, beta, maximizar)
                solucion_hormiga.append(tarea_seleccionada)
                recursos_disponibles.remove(tarea_seleccionada)
            actualizar_feromonas(solucion_hormiga, feromonas, maximizar)  # Actualizar feromonas
    # Devolver las feromonas después de las iteraciones
    return feromonas

# Ejemplo de uso del algoritmo de colonia de hormigas para maximización
num_iteraciones = 100
alfa = 1
beta = 1
feromonas_finales_max = colonia_de_hormigas(num_iteraciones, alfa, beta, maximizar=True)
print("Feromonas finales para maximización:")
for i, matriz_hormiga in enumerate(feromonas_finales_max):
    print("Hormiga", i+1, ":")
    for j, fila in enumerate(matriz_hormiga):
        print("Recurso", j+1, ":", fila)

# Ejemplo de uso del algoritmo de colonia de hormigas para minimización
feromonas_finales_min = colonia_de_hormigas(num_iteraciones, alfa, beta, maximizar=False)
print("\nFeromonas finales para minimización:")
for i, matriz_hormiga in enumerate(feromonas_finales_min):
    print("Hormiga", i+1, ":")
    for j, fila in enumerate(matriz_hormiga):
        print("Recurso", j+1, ":", fila)
