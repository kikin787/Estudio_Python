'''
Crear una clase cohce con los atributos:
marca, modelo, matricula, km
con los metodos:
init como constructor
avanzar(km) es aumenta
el valor de km en la cantidad
'''

class Carro:
    def __init__(self, marca, modelo, matricula, km):
        self.marca = marca
        self.modelo = modelo
        self.matricula = matricula
        self.km = km
    
    def avanzar(self, km):
        self.km += km

coche1 = Carro('BMW', 'X5', 'ABC-123-22', 1900)
print(coche1.__dict__)
coche1.avanzar(3000)
print(coche1.__dict__)


        