'''
Crear una clase Persona y otra clase Estudiante
La clase persona tiene el atributo nombre y el metodo mostrar_nombre()
La clase Estudiante debe heredar de la clase persona y utlizar el metodo mostrar_nombre()
de la calse Persona
'''

class Persona:
    def __init__(self, nombre):
        self.nombre = nombre
    
    def mostrar_nombre(self):
        print(f'El nombre es {self.nombre}')

class Estudiante(Persona):
    def __init__(self, nombre):
        super().__init__(nombre)
    
    def mostrar(self):
        super().mostrar_nombre()

estudiante = Estudiante('Kikin')
estudiante.mostrar()