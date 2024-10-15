try: 
    n1 = int(input("Ingresa el primer numero: "))
except ValueError as ex:
    print("Ingrese un valor que corresponda")
    n1 = int(input(": "))
except NameError as e:
    print("ocurrió un error")