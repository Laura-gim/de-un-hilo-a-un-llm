# Hito 1 — Concurrencia en CPU (Versión Python)

**Materia:** Programación Concurrente y Paralela  
**Integrantes:** Laura & Gabriel  
**Repositorio:** [de-un-hilo-a-un-llm](https://github.com/Laura-gim/de-un-hilo-a-un-llm.git)

---

## 1. Objetivo y Diseño

El objetivo del Hito 1 es demostrar dominio de hilos y sincronización resolviendo un problema con paralelismo de datos claro: la **reducción (suma) de un vector masivo de números flotantes** ($N = 2^{24}$ elementos inicializados en $1.0$). 

Se implementaron tres componentes principales:
1. **`secuencial.py`**: Calcula la suma utilizando un único flujo de ejecución como línea base de tiempo ($T_{seq}$).
2. **`paralelo.py`**: Distribuye el trabajo en bloques de tamaño fijo ($65536$ elementos) utilizando un pool de hilos (`threading`). Para evitar condiciones de carrera al reclamar los bloques pendientes, se implementó una primitiva de sincronización explícita (`threading.Lock`) protegiendo el contador compartido.
3. **`benchmark.py`**: Automatiza las pruebas descartando una corrida de calentamiento y calculando la **mediana** de múltiples repeticiones para mitigar la fluctuación del sistema operativo, calculando el *speedup* medido $S(T) = T_{seq} / T_{paralelo}$.

---

## 2. El Desafío del GIL en Python y el Modo NumPy

A diferencia de C++ (donde los hilos nativos ejecutan código máquina en paralelo absoluto), CPython opera bajo el **GIL (Global Interpreter Lock)**, el cual restringe la ejecución simultánea de bytecode de Python. 
- **Modo Python Puro:** Si se usa la suma nativa de listas (`sum()`), los hilos compiten por el GIL y no se obtiene aceleración (el speedup se estanca en $\approx 1.0\text{x}$ o empeora por el *overhead* de cambio de contexto).
- **Modo NumPy:** NumPy libera el GIL internamente durante las operaciones vectorizadas en C/Fortran (como `np.sum`). Esto permite que los hilos de Python deleguen el trabajo pesado y escalen de acuerdo con los núcleos físicos disponibles.

---

## 3. Resultados de Rendimiento y Benchmark

Las pruebas se ejecutaron sobre un vector de $N = 16\,777\,216$ elementos ($2^{24}$), tomando la mediana de 5 repeticiones:

| Hilos ($T$) | Tiempo Mediano (ms) [NumPy] | Speedup Medido ($S$) | Ley de Amdahl Teórica | Estado Checksum |
| :---: | :---: | :---: | :---: | :---: |
| **1** | $11.720\text{ ms}$ | $1.00\text{x}$ | $1.00\text{x}$ | OK |
| **2** | $7.150\text{ ms}$ | $1.64\text{x}$ | $1.75\text{x}$ | OK |
| **4** | $7.080\text{ ms}$ | $1.65\text{x}$ | $2.22\text{x}$ | OK |
| **8** | $7.120\text{ ms}$ | $1.64\text{x}$ | $2.50\text{x}$ | OK |

---

## 4. Análisis: ¿Por qué difieren el Speedup Medido y el Teórico?

1. **Límite de Núcleos Físicos (Sobre-suscripción):** En una CPU de 2 núcleos físicos (o máquina virtual estándar de pruebas), al superar los 2 hilos ($T > 2$), el sistema operativo debe alternar los hilos mediante *time-slicing*, generando un costo de cambio de contexto (*context switching*) que anula cualquier mejora adicional.
2. **Limitación de Ancho de Banda (*Memory-Bound*):** La suma de vectores no realiza un cómputo matemático complejo por cada elemento leído (bajo ratio de cómputo aritmético vs. accesos a memoria). Por lo tanto, el cuello de botella real pasa a ser el ancho de banda de la memoria RAM del sistema, el cual se satura rápidamente con pocos hilos.
3. **Overhead de Sincronización:** El uso de `threading.Lock` para el reparto dinámico introduce micro-bloqueos en la sección crítica, aunque al usar bloques grandes de $2^{16}$ elementos, este costo se amortiza eficientemente.

---

## 5. Instrucciones de Ejecución

Para probar el Hito 1 localmente:

```bash
# 1. Probar versión secuencial
python secuencial.py 16777216 numpy

# 2. Probar versión paralela con 4 hilos
python paralelo.py 16777216 4 numpy

# 3. Correr el benchmark completo
python benchmark.py 16777216 5 8 numpy