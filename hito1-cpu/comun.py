import sys
import numpy as np

SEMILLA = 42

def crear_vector(n, modo="numpy"):
    """
    Crea un vector de tamaño n con todos sus elementos en 1.0.
    La suma esperada exacta es float(n).
    """
    if modo == "numpy":
        np.random.seed(SEMILLA)
        # Usamos unos (1.0) tal como indica el diseño base para control estricto de precisión
        return np.ones(n, dtype=np.float32)
    else:
        # Modo Python puro (listas)
        return [1.0] * n

def validar_suma(resultado, n):
    """
    Valida que la suma obtenida coincida con N dentro de una tolerancia estricta.
    """
    esperado = float(n)
    tolerancia = 1e-6 * n
    diferencia = abs(resultado - esperado)
    return diferencia <= tolerancia, esperado, diferencia