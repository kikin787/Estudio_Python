import multiprocessing
import psutil


def suma_cuadrados(limite_inferior: int, limite_superior: int) -> None:
    valor = 0
    for i in range(limite_inferior, limite_superior+1):
        valor += i**2
    print(f"Suma de los cuadrados en el rango de {limite_inferior} a {limite_superior} = {valor}")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    limites_inferiores = range(0,10_000,100)
    limites_superiores = range(100,10_100,100)

    max_procesos = psutil.cpu_count(logical=False)
    lista_procesos = []

    for i in range(len(limites_inferiores)):
        p = multiprocessing.Process(target=suma_cuadrados, args=(limites_inferiores[i],limites_superiores[i]))
        lista_procesos.append(p)
        p.start()
        
        if len(lista_procesos)==max_procesos:
            for proc in lista_procesos:
                proc.join()
            lista_procesos = []

    for proc in lista_procesos:
        proc.join()
    print("Programa ha finalizado")