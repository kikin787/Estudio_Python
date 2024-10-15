try: 
    n1 = int(input("Ingresa el primer numero: "))
except Exception as e:
    print("Ocurrió un error")
else:
    print("No ocurrió ningun error")
finally:
    print("Se ejecuta siempre")