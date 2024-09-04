def clean_file(input_file, output_file):
    with open(input_file, 'r') as infile:
        with open(output_file, 'w') as outfile:
            for line in infile:
                # Eliminar caracteres no deseados: , () y espacio en blanco adicional
                line = line.replace(',', '').replace('(', '').replace(')', '').strip()
                # Omitir las líneas que contienen el formato "Capa #"
                if not line.startswith('Capa'):
                    outfile.write(line + '\n')

# Ejemplo de uso
input_file = 'C:\\Users\\Kikin\\Documents\\Python\\Estudio_Python\\MatrizCapas.txt'
output_file = 'C:\\Users\\Kikin\\Documents\\Python\\Estudio_Python\\MatrizCapas_limpio.txt'
clean_file(input_file, output_file)
print("Se ha limpiado el archivo correctamente.")
