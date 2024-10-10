class Perro:
    patas = 4
    
    def __init__(self, nombre, edad):
        self.name = nombre     
        self.age = edad   
    
    @classmethod
    def habla(cls):
        print("wow!")
    
    @classmethod
    def factory(cls):
        return cls("Peggy", 5)
    
    @classmethod
    def pedirPerros(cls, cantidad):
        perros = []
        for i in range(cantidad):
            nombre = input(f"Ingresa el nombre del perro {i+1}: ")
            edad = int(input(f"Ingresa la edad de {nombre}: "))
            perros.append(cls(nombre, edad))
        return perros
        
Perro.habla()

perro1 = Perro("Rocky", 2)
perro2 = Perro("Zeus", 3)
perro3 = Perro.factory()
print(perro3.age, perro3.name)

cantidad_perros = int(input("¿cuántos perros vas a crear?: "))
perros_creados = Perro.pedirPerros(cantidad_perros)

for perro in perros_creados:
    print(f"Perro: {perro.name}, Edad: {perro.age}")