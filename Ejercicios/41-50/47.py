"""
Hacer un menu de opciones que incluya la opcion de salir del programa
"""

while True:
    print("1. sumar")
    print("2. restar")
    print("3. salir")
    
    opcion = int(input("Elija una opción: "))

    if opcion == 1:
        print(f"La suma es {1+1}")
    elif opcion == 2:
        print(f"La resta es {2-1}")
    elif opcion == 3:
        break
    else:
        print("No es una opcion valida")