from io import open

#escritura
# texto = "Hola mundo"

# archivo = open("CursoPython/archivos/hola-mundo.txt", "w")
# archivo.write(texto)
# archivo.close()

#lectura
# archivo = open("CursoPython/archivos/hola-mundo.txt", "r")
# texto = archivo.read()
# archivo.close()
# print(texto)

#lectura como lista
# archivo = open("CursoPython/archivos/hola-mundo.txt", "r")
# texto = archivo.readlines()
# archivo.close()
# print(texto)

#metodos magicos
with open("CursoPython/archivos/hola-mundo.txt", "r") as archivo:
    print(archivo.readlines())