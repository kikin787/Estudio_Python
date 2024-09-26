# set significa grupo o conjunto
primer = {4, 1, 2, 1, 3, 2} # set elimina los duplicados
segundo = [3, 4, 5]
segundo = set(segundo)

print(primer | segundo) #unión de sets
print(primer & segundo) #intersección de sets
print(primer - segundo) #diferencia
print(primer ^ segundo) #diferencia simetrica 