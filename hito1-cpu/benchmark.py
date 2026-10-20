import time
import sys
import statistics
from comun import crear_vector, validar_suma
from secuencial import sumar_secuencial
from paralelo import sumar_paralelo

def benchmark(n, repeticiones=7, max_hilos=8, modo="numpy"):
    print(f"============================================================")
    print(f" HITO 1: Benchmark Concurrente en CPU (N = {n}, Modo = {modo})")
    print(f"============================================================")

    vector = crear_vector(n, modo)

    # 1. Medir Secuencial (tomando mediana de repeticiones)
    tiempos_seq = []
    for _ in range(repeticiones):
        _, t_seq = sumar_secuencial(vector, modo)
        tiempos_seq.append(t_seq)
    t_seq_mediana = statistics.median(tiempos_seq)
    print(f"-> Tiempo Secuencial Mediano (T_seq): {t_seq_mediana:.3f} ms\n")

    print(f"{'Hilos (T)':<10} | {'Mediana (ms)':<15} | {'Speedup Medido':<15} | {'Amdahl Teórico':<15} | {'Checksum':<10}")
    print("-" * 75)

    speedups_medidos = {}

    # 2. Medir Paralelo variando hilos
    for p in range(1, max_hilos + 1):
        # Calentamiento descartado
        sumar_paralelo(vector, p, tam_bloque=65536, modo=modo)

        tiempos_par = []
        for _ in range(repeticiones):
            res, t_par = sumar_paralelo(vector, p, tam_bloque=65536, modo=modo)
            tiempos_par.append(t_par)
        
        t_par_mediana = statistics.median(tiempos_par)
        speedup_medido = t_seq_mediana / t_par_mediana if t_par_mediana > 0 else 0.0
        speedups_medidos[p] = speedup_medido

        # Validar checksum con la última ejecución
        valido, _, _ = validar_suma(res, n)
        chk_str = "OK" if valido else "ERROR"

        # Ley de Amdahl teórica preliminar (asumiendo fracción secuencial estimada s = 0.25 para referencia general)
        # O despejada con el máximo hilo medido
        print(f"{p:<10} | {t_par_mediana:<15.3f} | {speedup_medido:<15.2f}x | {'-':<15} | {chk_str:<10}")

    print("-" * 75)
    print("Benchmark completado con éxito.")

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 16777216
    rep = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    hilos_max = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    modo = sys.argv[4] if len(sys.argv) > 4 else "numpy"

    benchmark(n, rep, hilos_max, modo)