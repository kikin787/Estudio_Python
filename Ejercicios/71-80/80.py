'''
Obtner la cantidad de memoria ram en mi computadora o laptop
pip install psutil
'''
import psutil

def obtener_ram():
    memoria = psutil.virtual_memory()
    memoria_total = memoria.total / (1024 ** 3)
    return memoria_total

mipc = obtener_ram()
print(f"El total de ram de la pc es de {round(mipc,2)}Gb")