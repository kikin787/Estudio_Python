'''
Crear una clase animal con los atributos especie y nombre
La clase debe tener los metodos:
init y hablar
el metodo hablar nos retorna una palabra
segun la interpretraciónd del sonido como un perro sería guau
'''

class Animal:
    def __init__(self, especie, nombre):
        self.especie = especie
        self.nombre = nombre

    def hablar(self):
        if self.especie.lower() == 'perro':
            return 'guau!!'
        elif self.especie.lower() == 'gato':
            return 'miau!!'

perro = Animal('Perro', 'Rocky')
gato = Animal('Gato', 'Cacho')

print(f'El primer animal es un {perro.especie} y se llama {perro.nombre}, también hace {perro.hablar()}')
print(f'El primer animal es un {gato.especie} y se llama {gato.nombre}, también hace {gato.hablar()}')