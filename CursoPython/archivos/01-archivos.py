from pathlib import Path
from time import ctime

archivo = Path('CursoPython/archivos/archivo-prueba.txt')

# archivo.exists()
# archivo.unlink()
# archivo.rename()

# try:
#     print(archivo.stat())
# except FileNotFoundError:
    
#     print(f"El archivo {archivo} no se encontró.")

print("acceso", ctime(archivo.stat().st_atime))
print("creación", ctime(archivo.stat().st_ctime))
print("modificación", ctime(archivo.stat().st_mtime))

