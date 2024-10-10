class Caminador:
    def caminar(self):
        print("caminando")

        
class Volador:
    def volar(self):
        print("volando")

class Nadador:
    def nadar(self):
        print("nadando")
        
class Atleta(Caminador, Nadador):
    def ejercicio(self):
        print("Hace ejercicio nadando")
