from abc import ABC, abstractmethod

class Model(ABC):
    @abstractmethod
    def guardar(self):
        pass
    
class Usuario(Model):
    def guardar(self):
        print("guardar en BBDD")
        
class Sesion(Model):
    def guardar(self):
        print("guardando en archivo")
        
def guardar(entidades):
    for entidad in entidades:
        entidad.guardar()

user = Usuario()
sesion = Sesion()

guardar([sesion, user])