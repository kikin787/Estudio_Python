'''
Crear una clase llamada Persona
Con los atributos: nombre, edad
*Un constructor, donde los datos pueden estar vacíos
*Los setters y getters
para cada uno de los atributos
*mostrar(): muestra los datos de la eprosna
*esMayor(): devuelve un valor lógico si es mayor de edad
'''

class Persona:
    def __init__(self, nombre=None, edad=None):
        self._nombre = nombre
        self._edad = edad
    
    @property
    def nombre(self):
        return self._nombre
    
    @nombre.setter
    def nombre(self, nuevo_nombre):
        self._nombre = nuevo_nombre

    @property
    def edad(self):
        return self._edad
    
    @edad.setter
    def edad(self, nueva_edad):
        if nueva_edad >= 0:
            self._edad = nueva_edad
        else:
            print('La edad no puede ser negativa')

    def mostrar(self):
        print(self.__dict__)

    def esMayor(self):
        if self._edad >= 18:
            print(f'La persona con nombre {self._nombre} es mayor de edad ya que tiene {self._edad}')
        else:
            print(f'La persona con nombre {self._nombre} es menor de edad ya que tiene {self._edad}')

persona1 = Persona('Kikin', 19)
persona1.esMayor()
persona1.mostrar()
persona1.nombre = 'Hola'
persona1.mostrar()
persona1.edad = -1


