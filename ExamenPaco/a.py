def ranks(arr):
    # Ordena el arreglo en orden descendente con índices
    sorted_arr = sorted(enumerate(arr), key=lambda x: x[1], reverse=True)
    # Crea el diccionario de rangos
    rank_dict = {}
    rank = 1
    for idx, (original_index, value) in enumerate(sorted_arr):
        if value not in rank_dict:
            rank_dict[value] = rank
            rank += 1
    
    return [rank_dict[val] for val in arr]

# Ejemplo de uso:
arr = [9, 3, 6, 10]
print(ranks(arr))