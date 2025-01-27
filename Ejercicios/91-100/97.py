"""
Hacer una función para crear un archivo de texto plano
"""

def crear_archivo(nombre_archivo):
    with open(nombre_archivo, 'w') as archivo:
        pass

crear_archivo('index.html')