'''
Escribir en un archivo html
hola que tal autodidacta
'''

def crear_archivo(nombre, contenido):
    with open(nombre, 'w') as archivo:
        archivo.write(contenido)

crear_archivo('index1.html', 'Hola amigos')