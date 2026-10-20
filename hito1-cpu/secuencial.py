import time
import sys
from comun import crear_vector, validar_suma

def sumar_secuencial(vector, modo="numpy"):
    inicio = time.perf_counter()
    if modo == "numpy":
        # np.sum en C libre del GIL
        total = float(np.sum(vector, dtype=np.float64))
    else:
        # Suma nativa de Python puro (atraviesa el GIL)
        total = float(sum(vector))
    fin = time.perf_counter()
    
    tiempo_ms = (fin - inicio) * 1000.0
    return total, tiempo_ms

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 16777216  # 2^24 por defecto
    modo = sys.argv[2] if len(sys.argv) > 2 else "numpy"

    vector = crear_vector(n, modo)
    total, tiempo = sumar_secuencial(vector, modo)
    valido, esperado, diff = validar_suma(total, n)

    print(f"[Secuencial] N={n}, Modo={modo}")
    print(f"  Tiempo: {tiempo:.3f} ms")
    print(f"  Resultado: {total} (Esperado: {esperado}, Diff: {diff:.2e})")
    print(f"  Checksum Válido: {'SÍ' if valido else 'NO'}")