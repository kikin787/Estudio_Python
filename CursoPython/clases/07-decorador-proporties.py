class Perro:
    def __init__(self, nombre):
        self.nombre = nombre
    
    @property
    def nombre(self):
        print("pasando por getter")
        return self.__nombre
    
    @nombre.setter
    def nombre(self, nombre):
        print("pasando por setter")
        if nombre.strip():
            self.__nombre = nombre
        return

        
perro = Perro("Kikin")
print(perro.nombre)