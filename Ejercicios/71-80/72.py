'''
Crear una clase ciculo con los siguientes atributos
* radio: radio del circulo
La clase debe tener los siguientes metodos:
*__init__ (self, radio): inicializa los atributos de la clase
* calcular el area_(self): calcula y devuelve el area del circulo
* calcular el perimetro(self): calcula y vuelve el perimetro del circulo
'''
import math
class Circulo:
    def __init__(self, radio):
        self.radio = radio
    
    def calcular_area(self):
        return math.pi * self.radio ** 2
    
    def calcular_perimetro(self):
        return 2 * math.pi * self.radio

circulo = Circulo(5)

print(f'El radio del circulo es {circulo.radio}. El área es: {round(circulo.calcular_area(), 2)}. El perimetro es: {round(circulo.calcular_perimetro(), 2)}')