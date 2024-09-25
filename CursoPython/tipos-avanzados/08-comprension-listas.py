usuarios = [
    ["Chanchito", 4], 
    ["Felipe", 1], 
    ["Pulga", 5]
]

# nombres = []
# for usuario in usuarios:
#     nombres.append(usuario[0])
# print(nombres)

# nombres = [usuario[0] for usuario in usuarios if usuario[1] == 5 ]
# print(nombres)

#filtar
# nombres = [usuario for usuario in usuarios if usuario[1] > 2]

# nombres = list(map(lambda user: user[0], usuarios))

menosUsuarios = list(filter(lambda user: user[1] > 2, usuarios))

print(menosUsuarios[1][0])