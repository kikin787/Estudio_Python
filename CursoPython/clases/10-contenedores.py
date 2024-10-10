class Producto:
    def __init__(self, nombre, precio):
        self.nombre = nombre
        self.precio = precio
    
    def __str__(self):
        return f"Producto: {self.nombre} - Precio: {self.precio}"
        
    
class Categoria:
    productos = []
    
    def __init__(self, nombre, productos):
        self.nombre = nombre
        self.productos = productos
    
    def agregar(self, producto):
        self.productos.append(producto)
        
    def imprimir(self):
        for producto in self.productos:
            print(producto)
            
bicileta = Producto("Bicicleta", 750)
kayak = Producto("Kayak", 1000)
skatebord = Producto("Skate", 1200)
deportes = Categoria("Deportes", [bicileta, kayak])
deportes.agregar(skatebord)
deportes.imprimir()