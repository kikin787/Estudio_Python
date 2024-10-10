class Perro:
    def __init__(self, nombre, edad):
        self.name = nombre     
        self.age = edad   
    def habla(self):
        print(f"{self.name} dice: wow! tiene la edad de {self.age} años")
        
mi_perro = Perro(edad = 15, nombre = "Kikin")
mi_perro.habla()
print(mi_perro.age)