import random

class AlgoritmoGenetico:
    def __init__(self, tam_poblacion, tam_cromosoma, rango_valores, elitismo, prob_mutacion=0.05, probabilidad_cruzamiento = 0.8):
        self.tam_poblacion = tam_poblacion
        self.tam_cromosoma = tam_cromosoma
        self.rango_valores = rango_valores
        self.mejor_solucion = None
        self.elitismo = elitismo
        self.prob_mutacion = prob_mutacion
        self.probabilidad_cruzamiento = probabilidad_cruzamiento

    def crear_cromosoma(self):
        cromosoma = self.rango_valores.copy()
        return random.shuffle(cromosoma)
    
    def crear_poblacion_inicial(self):
        return [self.crear_cromosoma() for _ in range(self.tam_poblacion)]

    def evaluar_fitness(self, cromosoma):
        # Implementa tu función de evaluación de aptitud según tu problema específico
        pass

    def seleccion(self):
        nueva_poblacion = []
        if self.elitismo > 0:
            self.poblacion.sort(key=lambda x: self.evaluar_fitness(x), reverse=True)
            elitismo_count = int(self.elitismo * self.tam_poblacion)
            elitismo = self.poblacion[:elitismo_count]
            nueva_poblacion = elitismo

        while len(nueva_poblacion) < self.tam_poblacion:
            padre1 = random.choice(self.poblacion)
            padre2 = random.choice(self.poblacion)
            if random.random() < self.probabilidad_cruzamiento:
                hijo1, hijo2 = self.cruzamiento(padre1, padre2)
                nueva_poblacion.append(hijo1)
                nueva_poblacion.append(hijo2)
            else:
                nueva_poblacion.append(padre1)
                nueva_poblacion.append(padre2)

        self.poblacion = nueva_poblacion[:self.tam_poblacion]

    def cruzamiento(self, padre1, padre2):
        punto_cruzamiento = random.randint(1, self.tam_cromosoma - 1)
        hijo1 = padre1[:punto_cruzamiento] + padre2[punto_cruzamiento:]
        hijo2 = padre2[:punto_cruzamiento] + padre1[punto_cruzamiento:]
        return hijo1, hijo2

    def mutacion(self):
        # Implementa tu método de mutación de cromosomas
        pass

    def entrenar(self, num_generaciones):
        self.crear_poblacion_inicial()
        for _ in range(num_generaciones):
            self.evaluar_fitness()
            self.seleccion()
            self.mutacion()
            
    def getElite(self, poblacio:list):
        elitismo_count = int(self.elitismo * self.tam_poblacion)
        return self.poblacion.sort(key=lambda x: self.evaluar_fitness(x), reverse=True)[:elitismo_count]

    def obtener_mejor_solucion(self):
        return self.mejor_solucion
