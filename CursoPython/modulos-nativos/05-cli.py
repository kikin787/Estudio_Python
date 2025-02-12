import os
from pathlib import Path
import sys

def cli(args):
    if len(args) == 1:
        print('no se pasaron argumentos')
        return
    if len(args) != 3:
        print("se necesitan dos argumentos")
        return
    
    origen = args[1]
    o = Path(origen)
    if not o.exists():
        print('Origen no existe')

    destino = args[2]
    d = Path(destino)
    if not d.exists():
        print('el destino no puede existir')

    os.rename(str(origen), str(destino))
    print("Archivo renombrado con exito")

cli(sys.argv)