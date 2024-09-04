animal = "    kikin EsCool   "
print(animal.upper()) #hace el string en minuscula
print(animal.lower()) #hace el string en mayuscula
print(animal.strip().capitalize()) #elimina los espacios tanto de izquierda como  derecha y hace la primer posición en mayuscula
print(animal.title()) #pone las primeras letras en mayuscula de cada palabra
print(animal.strip()) #elimina los espacios de izquierda y derecha
print(animal.lstrip()) #elimina los espacios de izquierda
print(animal.rstrip()) #elimina los espacios de derecha
print(animal.find("k")) #busca la primer posición de la letra
print(animal.replace(" ", "p")) #remplaza los espacios por la letra p
print(animal.lower().replace("k", "qu")) #hace el string en minuscula y remplaza la k por qu
print("ki" in animal) #devuelve true o false depende si está en el string
print("ki" not in animal)#devuelve true o false depende si no está en el string