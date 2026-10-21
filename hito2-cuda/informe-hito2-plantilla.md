# Informe Hito 2 — Kernels CUDA optimizados

**Programación Concurrente y Paralela · Grupo: Laura & Gabriel · Código: `hito2-cuda/`**
*(Completar los `COMPLETAR` con los datos medidos; borrar esta línea. Máximo 2 páginas.)*

## 1. Problema y diseño
Reducción (suma) de un vector de N `float32` (todos 1.0, suma esperada N), el mismo problema del Hito 1, portado a GPU con Numba CUDA. **Ingenuo:** cada hilo hace `atomicAdd` sobre un acumulador global único. **Optimizado:** carga coalescida a memoria compartida, reducción en árbol con `__syncthreads` y un parcial por bloque; los parciales se suman en el host en `float64`.

## 2. Cómo se midió
GPU: COMPLETAR (compute capability COMPLETAR) · CPU de Colab: COMPLETAR · Python COMPLETAR · Numba COMPLETAR. Tiempo de kernel con `cuda.event`; calentamiento descartado (compilación JIT); mediana de COMPLETAR repeticiones. Comando: `python benchmark_gpu.py --reps COMPLETAR`. Ancho de banda efectivo = 4·N bytes / tiempo. Validación: suma = N con tolerancia 1e-6·N.

## 3. Resultados
![Tiempo y ancho de banda](speedup_gpu.png)

| N | CPU (ms) | GPU ingenua (ms) | GPU optimizada (ms) | opt/ingenua | opt/CPU | GB/s opt | H2D (ms) |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| 2^20 | | | | | | | |
| 2^22 | | | | | | | |
| 2^24 | | | | | | | |

Datos de perfilado (Nsight Compute) o limitación declarada: COMPLETAR (tiempo, ocupación, throughput de memoria para ambos kernels).

## 4. Por qué mejora la versión optimizada (y qué no mejoró)
- **Atomic serializado:** COMPLETAR (cuántas veces más lento, con los datos).
- **Memoria compartida + árbol + coalescencia:** COMPLETAR (GB/s alcanzados vs. pico de la GPU; la operación es memory-bound).
- **CPU vs. GPU incluyendo la copia H2D:** COMPLETAR (¿conviene copiar el vector para sumarlo?).
- **Tamaño de bloque (128/256/512):** COMPLETAR.
- **Precisión float32 con N > 2^24:** COMPLETAR (¿saturó el acumulador del ingenuo?).

## 5. Conclusión
COMPLETAR (2–3 oraciones, solo lo que muestran los datos). Limitaciones: GPU compartida de Colab, variabilidad entre sesiones, COMPLETAR.

## 6. Conexión con la teoría
Modelo SIMT y warps (Clase 5); jerarquía de memoria, memoria compartida y coalescencia (Clase 6); patrón de reducción y perfilado (Clase 7).
