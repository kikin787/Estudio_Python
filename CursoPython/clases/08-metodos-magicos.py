class Perro:
    def __init__(self, nombre, edad):
        self.nombre = nombre
        self.edad = edad
        
    def __del__(self):
        print(f"Chao perro :( {self.nombre}")
        
    def __str__(self):
        return f"Clase Perro: {self.nombre}"
        
    def habla(self):
        print(f"{self.nombre} dice: wow")
        
perro = Perro("Kikin", 9)
del perro