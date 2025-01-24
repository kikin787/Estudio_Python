'''
Crear una clase persona con los atributos:
*** nombre, edad, dni
Con los metodos:
init()
es_mayor de edad() este retorna True si es mayor de edad
'''

class Persona():
    def __init__(self, nombre, edad, dni):
        self.nombre = nombre
        self.edad = edad
        self.dni = dni
    
    def esMayor(self):
        if self.edad >= 18:
            return True
        else:
            return False
kikin = Persona('Kikin', 18, 'Querty')

print(f'El nombre es: {kikin.nombre} y su edad es: {kikin.edad}')
if kikin.esMayor():
    print("Es mayor de edad")