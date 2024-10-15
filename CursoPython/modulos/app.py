# from usuarios import acciones
#acciones.guardar()

#import usuarios.acciones
#usuarios.acciones.guardar()

# import usuarios.gestion
# import usuarios.impuestos
from usuarios.impuestos.utilidades import pagar_impuestos
# import usuarios
pagar_impuestos()
print(__name__)
# print(usuarios.gestion.__name__)
# print(usuarios.impuestos.__package__)
# print(usuarios.gestion.__path__)
# print(usuarios.impuestos.__file__)
