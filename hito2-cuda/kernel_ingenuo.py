"""Hito 2 - Reduccion ingenua en GPU (atomicAdd sobre un unico acumulador global).

Uso:  python kernel_ingenuo.py [N] [repeticiones]
"""
import sys

import numpy as np
from numba import cuda

from comun import crear_vector, validar_suma, describir_entorno
from kernels import suma_atomica, poner_cero, cantidad_bloques, medir_ms, THREADS_POR_DEFECTO


def sumar_ingenuo(d_x, d_out, n, threads=THREADS_POR_DEFECTO):
    """Lanza el kernel y devuelve la suma (copia 1 float de vuelta al host)."""
    poner_cero[1, 1](d_out)
    suma_atomica[cantidad_bloques(n, threads), threads](d_x, d_out, n)
    return float(d_out.copy_to_host()[0])


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 16777216
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 7
    threads = THREADS_POR_DEFECTO
    bloques = cantidad_bloques(n, threads)

    x = crear_vector(n)
    d_x = cuda.to_device(x)
    d_out = cuda.to_device(np.zeros(1, dtype=np.float32))

    ms = medir_ms(lambda: suma_atomica[bloques, threads](d_x, d_out, n),
                  preparar=lambda: poner_cero[1, 1](d_out), reps=reps)
    total = sumar_ingenuo(d_x, d_out, n, threads)
    valido, esperado, diff = validar_suma(total, n)

    print(f"[GPU Ingenuo] N={n}, Bloques={bloques}, Hilos/bloque={threads}")
    print(f"  {describir_entorno()}")
    print(f"  Tiempo kernel (mediana de {reps}): {ms:.3f} ms")
    print(f"  Resultado: {total} (Esperado: {esperado}, Diff: {diff:.2e})")
    print(f"  Checksum Válido: {'SÍ' if valido else 'NO'}")
    sys.exit(0 if valido else 1)
