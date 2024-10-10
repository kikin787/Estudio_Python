class Perro:
    patas = 4
    
    def __init__(self, nombre, edad):
        self.name = nombre     
        self.age = edad   
    def habla(self):
        print(f"{self.name} dice: wow! tiene la edad de {self.age} años")


Perro.patas = 3
mi_perro = Perro(edad = 15, nombre = "Kikin")
mi_perro.patas = 5
mi_perro2 = Perro(edad = 15, nombre = "Kikin")

print(Perro.patas)
print(mi_perro.patas)
print(mi_perro2.patas)