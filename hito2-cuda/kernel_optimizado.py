"""Hito 2 - Reduccion optimizada en GPU (memoria compartida + arbol, un parcial por bloque).

Uso:  python kernel_optimizado.py [N] [repeticiones] [hilos_por_bloque]
"""
import sys

import numpy as np
from numba import cuda

from comun import crear_vector, validar_suma, describir_entorno
from kernels import crear_reduccion_shared, cantidad_bloques, medir_ms, THREADS_POR_DEFECTO


def sumar_optimizado(kernel, d_x, d_parcial, n, threads):
    """Lanza el kernel, copia los parciales (uno por bloque) y los suma en el host en float64."""
    kernel[cantidad_bloques(n, threads), threads](d_x, d_parcial, n)
    return float(d_parcial.copy_to_host().sum(dtype=np.float64))


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 16777216
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 7
    threads = int(sys.argv[3]) if len(sys.argv) > 3 else THREADS_POR_DEFECTO
    bloques = cantidad_bloques(n, threads)

    kernel = crear_reduccion_shared(threads)
    x = crear_vector(n)
    d_x = cuda.to_device(x)
    d_parcial = cuda.device_array(bloques, dtype=np.float32)

    ms = medir_ms(lambda: kernel[bloques, threads](d_x, d_parcial, n), reps=reps)
    total = sumar_optimizado(kernel, d_x, d_parcial, n, threads)
    valido, esperado, diff = validar_suma(total, n)

    print(f"[GPU Optimizado] N={n}, Bloques={bloques}, Hilos/bloque={threads}")
    print(f"  {describir_entorno()}")
    print(f"  Tiempo kernel (mediana de {reps}): {ms:.3f} ms")
    print(f"  Resultado: {total} (Esperado: {esperado}, Diff: {diff:.2e})")
    print(f"  Checksum Válido: {'SÍ' if valido else 'NO'}")
    sys.exit(0 if valido else 1)
