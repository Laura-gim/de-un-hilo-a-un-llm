import time
import sys
import threading
import numpy as np
from comun import crear_vector, validar_suma

def worker_bloque(vector, inicio_bloque, fin_bloque, modo, id_hilo, resultados, idx):
    """
    Cada hilo procesa una porción asignada del vector.
    """
    if modo == "numpy":
        sub_suma = float(np.sum(vector[inicio_bloque:fin_bloque], dtype=np.float64))
    else:
        sub_suma = float(sum(vector[inicio_bloque:fin_bloque]))
    resultados[idx] = sub_suma

def sumar_paralelo(vector, num_hilos, tam_bloque=65536, modo="numpy"):
    n = len(vector)
    resultados = [0.0] * ((n + tam_bloque - 1) // tam_bloque)
    
    # Primitiva de sincronización: Lock para manejar el índice compartido de bloques
    lock = threading.Lock()
    proximo_bloque = [0]
    total_bloques = len(resultados)

    def tarea_hilo():
        while True:
            with lock:
                if proximo_bloque[0] >= total_bloques:
                    return
                b_idx = proximo_bloque[0]
                proximo_bloque[0] += 1
            
            inicio = b_idx * tam_bloque
            min_fin = min(inicio + tam_bloque, n)
            worker_bloque(vector, inicio, min_fin, modo, 0, resultados, b_idx)

    inicio_tiempo = time.perf_counter()
    
    hilos = []
    for _ in range(num_hilos):
        h = threading.Thread(target=tarea_hilo)
        hilos.append(h)
        h.start()

    for h in hilos:
        h.join()

    total = sum(resultados)
    fin_tiempo = time.perf_counter()

    tiempo_ms = (fin_tiempo - inicio_tiempo) * 1000.0
    return total, tiempo_ms

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 16777216
    num_hilos = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    modo = sys.argv[3] if len(sys.argv) > 3 else "numpy"

    vector = crear_vector(n, modo)
    total, tiempo = sumar_paralelo(vector, num_hilos, tam_bloque=65536, modo=modo)
    valido, esperado, diff = validar_suma(total, n)

    print(f"[Paralelo] N={n}, Hilos={num_hilos}, Modo={modo}")
    print(f"  Tiempo: {tiempo:.3f} ms")
    print(f"  Resultado: {total} (Esperado: {esperado}, Diff: {diff:.2e})")
    print(f"  Checksum Válido: {'SÍ' if valido else 'NO'}")