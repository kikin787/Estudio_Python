punto = { "x": 25,
          "y": 50
         }
# print(punto)
# print(punto["x"])
# print(punto["y"])

# punto["z"] = 45
# print(punto)

# if "p" in punto:
#     print(punto["p"])

# print(punto.get("x"))
# print(punto.get("p", 97))
# del punto["x"]
# del (punto["y"])

# print(punto)
# punto["x"] = 25

for valor in punto:
    print(valor, punto[valor])
    
for valor in punto.items():
    print(valor)

for llave, valor in punto.items():
    print(llave, valor)
    
usuarios = [
    {"id": 1, "nombre": "Kike"},
    {"id": 2, "nombre": "Fátima"},
    {"id": 3, "nombre": "Enrique"},
    {"id": 4, "nombre": "Felipe"},
]

tuple(usuarios)
print(usuarios)
# for usuario in usuarios:
#     print(usuario["nombre"])