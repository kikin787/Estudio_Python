class Animal:
    def comer(self):
        print("comiendo")
        

class Perro(Animal):
    def pasear(self):
        print("paseando")
        
perro = Perro()
perro.comer()
    
        
class Kikin(Perro):
    def programar(self):
        print("programando")

kikin = Kikin()
kikin.comer()