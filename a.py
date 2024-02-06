import pygame  # Importa la biblioteca Pygame para la interfaz gráfica
import sys  # Importa sys para interactuar con el sistema
import numpy as np  # Importa numpy para operaciones numéricas eficientes

# Inicialización de Pygame
pygame.init()

# Parámetros de la simulación
tamano_celda = 10  # Tamaño de la celda en la cuadrícula
tamano_mundo = 60  # Tamaño del mundo bidimensional
probabilidad_enfermarse = 5/100000  # Probabilidad inicial de contraer la enfermedad
duracion_enfermo = 14  # Duración total de la enfermedad en días
dias_contagioso = 7  # Número de días en que un individuo es contagioso
prob_contagio_primeros_dias = 0.7  # Probabilidad de contagio durante los primeros días de la enfermedad
prob_contagio_siguientes_dias = 0.3  # Probabilidad de contagio en días posteriores al período contagioso inicial
prob_recuperacion_despues_dias_contagioso = 0.6  # Probabilidad de recuperación después del período contagioso
prob_muerte = 0.000075  # Probabilidad de muerte debido a la enfermedad
duracion_inmunidad = 120  # Duración de la inmunidad después de la recuperación
prob_vacio = 0.1  # Probabilidad de tener un espacio vacío en la cuadrícula

# Definición de los posibles estados
SANO = 0
ENFERMO = 1
RECUPERADO = 2
MUERTO = 3
VACIO = 4

# Inicialización del mundo y matrices de tiempo
mundo = np.zeros((tamano_mundo, tamano_mundo), dtype=int)  # Matriz que representa el estado del mundo
tiempo_enfermo = np.zeros((tamano_mundo, tamano_mundo), dtype=int)  # Matriz para controlar el tiempo que llevan enfermos
tiempo_inmune = np.zeros((tamano_mundo, tamano_mundo), dtype=int)  # Matriz para controlar el tiempo de inmunidad

# Variable para contar los días transcurridos
dias_transcurridos = 0

# Configuración de la pantalla de la simulación
ancho_pantalla = tamano_mundo * tamano_celda + 200  # Ancho de la ventana, se agrega espacio para información lateral
alto_pantalla = tamano_mundo * tamano_celda  # Altura de la ventana
pantalla = pygame.display.set_mode((ancho_pantalla, alto_pantalla))  # Crea la ventana
pygame.display.set_caption('Simulación de Epidemia')  # Establece el título de la ventana

# Función para inicializar un individuo enfermo
def iniciar_enfermedad():
    return np.random.rand() < probabilidad_enfermarse

# Función para inicializar espacios vacíos aleatorios
def inicializar_vacios():
    for i in range(tamano_mundo):
        for j in range(tamano_mundo):
            if np.random.rand() < prob_vacio:
                mundo[i, j] = VACIO

# Función para contar vecinos enfermos con frontera toroidal
def contar_vecinos_enfermos_toroidal(mundo, i, j):
    return np.sum(mundo[(i-1) % tamano_mundo:(i+2) % tamano_mundo, (j-1) % tamano_mundo:(j+2) % tamano_mundo] == ENFERMO)

# Función para actualizar el estado del mundo en cada paso con frontera toroidal
def actualizar_mundo_toroidal():
    global dias_transcurridos  # Declarar que estamos utilizando la variable global

    nuevo_mundo = mundo.copy()
    nuevo_tiempo_enfermo = tiempo_enfermo.copy()
    nuevo_tiempo_inmune = tiempo_inmune.copy()

    # Iterar sobre cada celda en el mundo
    for i in range(tamano_mundo):
        for j in range(tamano_mundo):
            if mundo[i, j] == SANO:  # Si el individuo está sano
                vecinos_enfermos = contar_vecinos_enfermos_toroidal(mundo, i, j)

                # Verificar si un individuo sano se enferma
                if np.random.rand() < probabilidad_enfermarse * (1 - vecinos_enfermos / 8):
                    nuevo_mundo[i, j] = ENFERMO  # Cambiar a estado ENFERMO
                    nuevo_tiempo_enfermo[i, j] = 0  # Reiniciar tiempo enfermo

            elif mundo[i, j] == ENFERMO:  # Si el individuo está enfermo
                # Incrementar el tiempo que llevan enfermos
                nuevo_tiempo_enfermo[i, j] += 1

                # Definir la probabilidad de contagio dependiendo de los días transcurridos
                if nuevo_tiempo_enfermo[i, j] <= dias_contagioso:
                    prob_contagio = prob_contagio_primeros_dias
                else:
                    prob_contagio = prob_contagio_siguientes_dias

                # Propagar la enfermedad a individuos sanos cercanos con frontera toroidal
                for x in range(i-1, i+2):
                    for y in range(j-1, j+2):
                        if nuevo_mundo[x % tamano_mundo, y % tamano_mundo] == SANO and np.random.rand() < prob_contagio:
                            nuevo_mundo[x % tamano_mundo, y % tamano_mundo] = ENFERMO
                            nuevo_tiempo_enfermo[x % tamano_mundo, y % tamano_mundo] = 0

                # Probabilidad de recuperarse después de ciertos días
                if nuevo_tiempo_enfermo[i, j] > dias_contagioso and np.random.rand() < prob_recuperacion_despues_dias_contagioso:
                    nuevo_mundo[i, j] = RECUPERADO
                    nuevo_tiempo_enfermo[i, j] = 0
                    nuevo_tiempo_inmune[i, j] = duracion_inmunidad

                # Probabilidad de muerte
                if np.random.rand() < prob_muerte:
                    nuevo_mundo[i, j] = MUERTO

            elif mundo[i, j] == RECUPERADO:  # Si el individuo está recuperado
                nuevo_tiempo_inmune[i, j] -= 1  # Disminuir el tiempo de inmunidad

                # Volver a ser susceptible después de la inmunidad
                if nuevo_tiempo_inmune[i, j] <= 0:
                    nuevo_mundo[i, j] = SANO  # Cambiar a estado SANO

    dias_transcurridos += 1  # Aumentar el contador de días

    return nuevo_mundo, nuevo_tiempo_enfermo, nuevo_tiempo_inmune

# Función para mostrar información lateral
def mostrar_info_lateral():
    # Creación de textos con la información de los estados y días
    fuente = pygame.font.Font(None, 24)
    texto_sano = fuente.render('Sano', True, (255, 255, 255))
    texto_enfermo = fuente.render('Enfermo', True, (255, 0, 0))
    texto_recuperado = fuente.render('Recuperado', True, (0, 255, 0))
    texto_muerto = fuente.render('Muerto', True, (156, 156, 156))
    texto_vacio = fuente.render('Vacío', True, (0, 0, 0))
    texto_dias = fuente.render(f'Días: {dias_transcurridos}', True, (0, 0, 0))

    # Mostrar los textos en la pantalla
    pantalla.blit(texto_sano, (ancho_pantalla - 180, 50))
    pantalla.blit(texto_enfermo, (ancho_pantalla - 180, 100))
    pantalla.blit(texto_recuperado, (ancho_pantalla - 180, 150))
    pantalla.blit(texto_muerto, (ancho_pantalla - 180, 200))
    pantalla.blit(texto_vacio, (ancho_pantalla - 180, 250))
    pantalla.blit(texto_dias, (ancho_pantalla - 180, 300))

# Bucle principal de la simulación
ejecutando = True
reloj = pygame.time.Clock()

# Inicializar espacios vacíos y algunos individuos enfermos
inicializar_vacios()
for i in range(tamano_mundo):
    for j in range(tamano_mundo):
        if mundo[i, j] != VACIO:
            mundo[i, j] = ENFERMO if iniciar_enfermedad() else SANO

# Bucle principal de la simulación
while ejecutando:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            ejecutando = False  # Salir del bucle si se cierra la ventana

    # Actualizar el mundo
    mundo, tiempo_enfermo, tiempo_inmune = actualizar_mundo_toroidal()

    # Limpiar la pantalla
    pantalla.fill((255, 255, 255))  # Fondo blanco para el área principal

    # Dibujar la barra lateral
    pantalla.fill((200, 200, 200), pygame.Rect(ancho_pantalla - 200, 0, 200, alto_pantalla))

    # Dibujar la cuadrícula
    for x in range(0, ancho_pantalla, tamano_celda):
        pygame.draw.line(pantalla, (200, 200, 200), (x, 0), (x, alto_pantalla))
    for y in range(0, alto_pantalla, tamano_celda):
        pygame.draw.line(pantalla, (200, 200, 200), (0, y), (ancho_pantalla, y))

    # Dibujar el estado actual del mundo
    for i in range(tamano_mundo):
        for j in range(tamano_mundo):
            if mundo[i, j] == ENFERMO:
                pygame.draw.rect(pantalla, (255, 0, 0), (j * tamano_celda, i * tamano_celda, tamano_celda, tamano_celda))
            elif mundo[i, j] == RECUPERADO:
                pygame.draw.rect(pantalla, (0, 255, 0), (j * tamano_celda, i * tamano_celda, tamano_celda, tamano_celda))
            elif mundo[i, j] == MUERTO:
                pygame.draw.rect(pantalla, (100, 100, 100), (j * tamano_celda, i * tamano_celda, tamano_celda, tamano_celda))
            elif mundo[i, j] == VACIO:
                pygame.draw.rect(pantalla, (0, 0, 0), (j * tamano_celda, i * tamano_celda, tamano_celda, tamano_celda))

    # Mostrar información lateral
    mostrar_info_lateral()

    # Actualizar la pantalla
    pygame.display.flip()

    # Controlar la velocidad de la simulación
    reloj.tick(35)  # Ajusta esta velocidad según tus necesidades

# Salir de Pygame y finalizar el programa
pygame.quit()
sys.exit()