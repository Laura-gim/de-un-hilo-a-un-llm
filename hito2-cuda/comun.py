"""Datos y utilidades compartidas por los scripts del Hito 2.

Mismo problema que el Hito 1: reduccion (suma) de un vector de N float32 con todos
los elementos en 1.0, de modo que la suma esperada es exactamente N.
"""
import os

import numpy as np

SEMILLA = 42

# Si NUMBA_ENABLE_CUDASIM=1, Numba ejecuta los kernels en un SIMULADOR en CPU
# (sirve para probar la logica sin GPU; los tiempos NO representan a la GPU).
SIMULADOR = os.environ.get("NUMBA_ENABLE_CUDASIM") == "1"


def crear_vector(n):
    """Vector de n elementos float32, todos 1.0 (igual que el Hito 1, modo numpy)."""
    np.random.seed(SEMILLA)
    return np.ones(n, dtype=np.float32)


def validar_suma(resultado, n):
    """Valida que la suma coincida con N dentro de una tolerancia estricta (igual que el Hito 1)."""
    esperado = float(n)
    tolerancia = 1e-6 * n
    diferencia = abs(resultado - esperado)
    return diferencia <= tolerancia, esperado, diferencia


def describir_entorno():
    """Texto con la GPU y la version de Numba, para declarar el hardware en el informe."""
    try:
        import numba
        from numba import cuda
        if SIMULADOR:
            return f"SIMULADOR CUDA de Numba {numba.__version__} (sin GPU real)"
        dev = cuda.get_current_device()
        nombre = dev.name.decode() if isinstance(dev.name, bytes) else dev.name
        cc = ".".join(map(str, dev.compute_capability))
        return f"GPU: {nombre} (compute capability {cc}), Numba {numba.__version__}"
    except Exception as e:  # noqa: BLE001
        return f"No se pudo leer la GPU: {e}"
