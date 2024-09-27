lista = (1, 2, 3, 4)
# # print(lista)
# # print(*lista)

# lista2 = [5, 6]

# combinada = ["hola",*lista, "mundo", *lista2]
# print(*combinada)

punto1 = {"x": 19, "y": "hola"}
punto2 = {"y": 5}

nuevoPunto = {**punto1, "lala": "hola", **punto2, "z": "mundo"}
print(nuevoPunto)