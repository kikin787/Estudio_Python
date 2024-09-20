resultado = ""

while True:
    print("Las operaciones son suma, multi, div y resta");
    if not resultado:
        resultado = input("Ingresa el primer número: ")
        if resultado.lower() == "salir":
            break
        resultado = int(resultado);
    op = input("Ingresa operación: ")
    if op.lower() == "salir":
        break
    n2 = input("Ingresa el siguiente número: ")
    if n2.lower() == "salir":
        break
    n2 = int(n2)
    
    if op.lower() == "sum":
        resultado += n2
    elif op.lower() == "multi":
        resultado *= n2
    elif op.lower() == "div":
        resultado /= n2
    elif op.lower() == "resta":
        resultado -= n2
    else:
        print("Operación no válida")
        break
    
    print(f"El resultado : {resultado}" )
    
        