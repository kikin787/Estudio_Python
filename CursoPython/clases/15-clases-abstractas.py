from abc import ABC, abstractmethod


class Model(ABC):
    @property
    @abstractmethod
    def tabla(self):
        pass
    @abstractmethod
    def guardar(self):
        pass        
    
    @classmethod
    def buscar_por_id(cls, _id):
        print(f"Buscando por id {_id} en la tabla {cls.tabla}")
        
class Usuario(Model):
    tabla = "Usuario"
    
    def guardar(self):
        print("guardandno usuario")
    
usuario = Usuario()
Usuario.buscar_por_id(123)
usuario.guardar()