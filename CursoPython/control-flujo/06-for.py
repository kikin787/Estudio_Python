str = "hoa como estas"

# for i in range(len(str)):
#     print(str[i])

# for numero in range(5):
#     print(numero + 1 + 5, numero * "hola")

buscar = 10
for numero in range(5):
    print(numero)
    if numero == buscar:
        print("encontrado", buscar)
        break;
else:
    print("no hay numero hijo")
    
for char in str:
    print(char)