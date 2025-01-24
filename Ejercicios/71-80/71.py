"""
Crear una clase Rectangulo que los siguientes atributos
base: base del rectangulo
altura: altura del rectangulo
La clase debe de tener los siguientes metodos:
** __init__(self, base, altura): inicializa con los atributos de la clase
** calcular_area(self): calcula y devuelve el area del rectangulo
** calcular_perimetro(self): calcula y devuelve el perimetro del rectangulo
"""

class Rectangulo:
    def __init__(self, base, altura):
        self.base = base
        self.altura = altura
    
    def calcular_area(self):
        return self.base * self.altura
    
    def calcular_perimetro(self):
        return 2 * self.base + 2 * self.altura
    
rectangulo1 = Rectangulo(6,7)
rectangulo2 = Rectangulo(8,9)
rectangulo3 = Rectangulo(1,3)

print(f"""El área del rectangulo con base de {rectangulo1.base}
    y altura de {rectangulo1.altura} es: {rectangulo1.calcular_area()}. 
    Y su perimetro es: {rectangulo1.calcular_perimetro()}""")
print(f"""El área del rectangulo con base de {rectangulo2.base}
    y altura de {rectangulo2.altura} es: {rectangulo2.calcular_area()}. 
    Y su perimetro es: {rectangulo2.calcular_perimetro()}""")
print(f"""El área del rectangulo con base de {rectangulo3.base} 
    y altura de {rectangulo3.altura} es: {rectangulo3.calcular_area()}. 
    Y su perimetro es: {rectangulo3.calcular_perimetro()}""")


