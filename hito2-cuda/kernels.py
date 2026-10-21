"""Kernels CUDA (Numba) del Hito 2: reduccion (suma) de un vector float32.

  - suma_atomica:       version INGENUA. Cada hilo hace atomicAdd sobre un unico acumulador
                        en memoria global. Correcta, pero todos compiten por la misma direccion.
  - reduce_shared:      version OPTIMIZADA. Cada bloque carga sus elementos a MEMORIA COMPARTIDA
                        (lecturas coalescidas) y los reduce en ARBOL; produce un parcial por bloque.
"""
import statistics
import time

from numba import cuda, float32

from comun import SIMULADOR

THREADS_POR_DEFECTO = 256


# ---------------------------------------------------------------- version ingenua
@cuda.jit
def poner_cero(out):
    out[0] = 0.0


@cuda.jit
def suma_atomica(x, out, n):
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if i < n:
        cuda.atomic.add(out, 0, x[i])


# ---------------------------------------------------------------- version optimizada
def crear_reduccion_shared(threads=THREADS_POR_DEFECTO):
    """Devuelve el kernel de reduccion para un tamano de bloque `threads` (potencia de 2).

    El tamano de la memoria compartida tiene que ser una constante de compilacion; por eso el
    kernel se crea en una funcion y `threads` queda fijo dentro de el."""
    if threads < 2 or threads & (threads - 1):
        raise ValueError("threads tiene que ser una potencia de 2")

    @cuda.jit
    def reduce_shared(x, parcial, n):
        s = cuda.shared.array(threads, float32)        # memoria compartida del bloque
        tid = cuda.threadIdx.x
        i = cuda.blockIdx.x * cuda.blockDim.x + tid

        # 1) carga coalescida: el hilo i lee x[i]; los hilos de borde (i >= n) aportan 0
        if i < n:
            s[tid] = x[i]
        else:
            s[tid] = float32(0.0)
        cuda.syncthreads()

        # 2) reduccion en arbol: en cada paso, la mitad de los hilos suma otra posicion
        stride = threads // 2
        while stride > 0:
            if tid < stride:
                s[tid] += s[tid + stride]
            cuda.syncthreads()
            stride //= 2

        # 3) el hilo 0 escribe el total del bloque
        if tid == 0:
            parcial[cuda.blockIdx.x] = s[0]

    return reduce_shared


def cantidad_bloques(n, threads):
    return (n + threads - 1) // threads


# ---------------------------------------------------------------- medicion
def medir_ms(lanzar, preparar=None, reps=7):
    """Mediana (ms) de `reps` lanzamientos, tras una corrida de calentamiento descartada.

    En GPU se usa cuda.event (mide solo el kernel, sin copias). La primera llamada incluye la
    compilacion JIT de Numba, por eso se descarta. `preparar` se ejecuta ANTES de iniciar el
    cronometro (por ejemplo, para poner el acumulador en cero)."""
    if preparar:
        preparar()
    lanzar()                                           # calentamiento (JIT + caches)
    cuda.synchronize()
    tiempos = []
    for _ in range(reps):
        if preparar:
            preparar()
        cuda.synchronize()
        if SIMULADOR:
            t0 = time.perf_counter()
            lanzar()
            cuda.synchronize()
            tiempos.append((time.perf_counter() - t0) * 1000.0)
        else:
            ini, fin = cuda.event(), cuda.event()
            ini.record()
            lanzar()
            fin.record()
            fin.synchronize()
            tiempos.append(cuda.event_elapsed_time(ini, fin))     # milisegundos
    return statistics.median(tiempos)
