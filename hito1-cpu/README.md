# Hito 1 — Concurrencia en CPU (Versión Python)

**Materia:** Programación Concurrente y Paralela  
**Integrantes:** Laura & Gabriel  
**Repositorio:** [de-un-hilo-a-un-llm](https://github.com/Laura-gim/de-un-hilo-a-un-llm.git)

---

## 1. Objetivo y Diseño

El objetivo del Hito 1 es demostrar dominio de hilos y sincronización resolviendo un problema con paralelismo de datos claro: la **reducción (suma) de un vector masivo de números flotantes** ($N = 2^{24} = 16\,777\,216$ elementos inicializados en $1.0$). 

Se implementaron tres componentes principales en Python:
1. **`secuencial.py`**: Calcula la suma utilizando un único flujo de ejecución como línea base de tiempo ($T_{seq}$).
2. **`paralelo.py`**: Distribuye el trabajo de forma dinámica en bloques utilizando hilos (`threading`). Para evitar condiciones de carrera al reclamar los bloques de trabajo pendientes, se implementó una primitiva de sincronización explícita mediante un **`threading.Lock`** protegiendo el índice compartido.
3. **`benchmark.py`**: Automatiza las pruebas descartando una corrida inicial de calentamiento (*warm-up*) y calculando la **mediana** de múltiples repeticiones para mitigar la fluctuación del sistema operativo, evaluando el *speedup* medido $S(T) = T_{seq} / T_{paralelo}$.

---

## 2. El Desafío del GIL en Python y el Modo NumPy

A diferencia de C++ (donde los hilos nativos ejecutan instrucciones en paralelo absoluto), CPython opera bajo el **GIL (Global Interpreter Lock)**, el cual restringe la ejecución simultánea de bytecode de Python puro. 
- **Modo Python Puro:** Si se operara sobre estructuras nativas (como `sum()` sobre listas), los hilos competirían por el GIL impidiendo cualquier aceleración real.
- **Modo NumPy:** NumPy libera el GIL internamente durante la ejecución de sus funciones vectorizadas altamente optimizadas (como `np.sum`), permitiendo delegar el cómputo pesado y aprovechar la concurrencia multihilo.

---

## 3. Resultados de Rendimiento y Benchmark

Las pruebas se ejecutaron sobre Google Colab utilizando un vector de $N = 16\,777\,216$ elementos en modo `numpy`, tomando la mediana de repeticiones (Tiempo secuencial de referencia $T_{seq} = 13.850\text{ ms}$):

| Hilos ($T$) | Mediana (ms) | Speedup Medido ($S$) | Amdahl Teórico | Estado Checksum |
| :---: | :---: | :---: | :---: | :---: |
| **1** | $16.264\text{ ms}$ | $0.85\text{x}$ | $0.85\text{x}$ | OK |
| **2** | $13.734\text{ ms}$ | $1.01\text{x}$ | $1.01\text{x}$ | OK |
| **3** | $14.637\text{ ms}$ | $0.95\text{x}$ | $0.98\text{x}$ | OK |
| **4** | $14.899\text{ ms}$ | $0.93\text{x}$ | $0.95\text{x}$ | OK |
| **5** | $14.655\text{ ms}$ | $0.95\text{x}$ | $0.97\text{x}$ | OK |
| **6** | $15.358\text{ ms}$ | $0.90\text{x}$ | $0.93\text{x}$ | OK |
| **7** | $15.452\text{ ms}$ | $0.90\text{x}$ | $0.93\text{x}$ | OK |
| **8** | $15.758\text{ ms}$ | $0.88\text{x}$ | $0.92\text{x}$ | OK |

---

## 4. Análisis: Rendimiento y Limitaciones del Sistema

1. **Operación *Memory-Bound* (Cuello de botella de memoria):** La suma de un vector masivo no representa un cómputo aritmético pesado por cada byte leído, sino que satura rápidamente el ancho de banda del bus de memoria RAM de la máquina virtual. Al superar el caudal soportado por la memoria, añadir más hilos deja de aportar mejoras.
2. **Impacto de la Tarea de Corta Duración:** Dado que el tiempo total de ejecución secuencial es sumamente bajo ($\approx 13.8\text{ ms}$), el *overhead* inherente al agendamiento de hilos del sistema operativo, el uso de la primitiva `threading.Lock` para el reparto dinámico y el rearmado de resultados supera o iguala el beneficio marginal de dividir la tarea.
3. **Escalabilidad y Sobre-suscripción:** Se observa un leve pico de mejora óptima al utilizar 2 hilos ($S = 1.01\text{x}$), estabilizándose y descendiendo levemente al superar los núcleos físicos reales de la instancia debido al costo de cambio de contexto (*context switching*).

---

## 5. Instrucciones de Ejecución

Para reproducir las pruebas localmente o en Google Colab:

```bash
# 1. Probar versión secuencial
python secuencial.py 16777216 numpy

# 2. Probar versión paralela con 4 hilos
python paralelo.py 16777216 4 numpy

# 3. Correr el benchmark completo
python benchmark.py 16777216 5 8 numpy