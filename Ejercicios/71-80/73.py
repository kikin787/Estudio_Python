'''
Crear una clase libro
atributos:
titulo, autor, editorial, año de la publicación
Metodos:
constructor para inicializar los atributos
'''

class Libro:
    def __init__(self, titulo, autor, editorial, año):
        self.titulo = titulo
        self.autor = autor
        self.editorial = editorial
        self.año = año

librito = Libro(
    'Cien años de soledad',
    'Enrique Vélez',
    'No sabo',
    '100'
)

print(librito.__dict__)