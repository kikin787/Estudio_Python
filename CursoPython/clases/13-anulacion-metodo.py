class Ave:
    def __init__(self):
        self.volador = "volador"
    def vuela(self):
        print("Vuela ave")

class Pato(Ave):
    def __init__(self):
        super().__init__()
        self.nada = "nadador"
    def vuela(self):
        print("Vuela pato")
        super().vuela()
        
pato = Pato()
pato.vuela()
print(pato.volador, pato.nada)