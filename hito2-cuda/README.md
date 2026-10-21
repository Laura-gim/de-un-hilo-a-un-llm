# Hito 2 — Kernels CUDA optimizados (Versión Python / Numba)

**Materia:** Programación Concurrente  
**Integrantes:** Laura Elizabeth Giménez y Oscar Gabriel Sosa  
**Repositorio:** [de-un-hilo-a-un-llm](https://github.com/Laura-gim/de-un-hilo-a-un-llm.git)

---

## 1. Objetivo y Diseño

Portar a GPU el problema del Hito 1 —la **reducción (suma) de un vector de $N$ `float32` con todos los elementos en $1.0$** (suma esperada exacta $= N$)— y **optimizarlo** con una técnica de las Clases 6–7. Se resuelve en Python con **Numba CUDA** (`@cuda.jit`), manteniendo el mismo problema, la misma validación y la misma forma de medir que el Hito 1.

Componentes:
1. **`kernel_ingenuo.py`** — *baseline*: cada hilo hace `cuda.atomic.add` sobre **un único acumulador** en memoria global. Es correcto pero lento, porque todos los hilos compiten por la misma dirección y las actualizaciones se serializan.
2. **`kernel_optimizado.py`** — cada bloque carga sus elementos a **memoria compartida** y los reduce en **árbol**; cada bloque produce **un parcial**, y los parciales se suman en el host en `float64`.
3. **`benchmark_gpu.py`** — compara **CPU secuencial (Hito 1) vs. GPU ingenua vs. GPU optimizada** para varios $N$, con calentamiento descartado y **mediana** de repeticiones, y barre el número de hilos por bloque.
4. **`kernels.py`**, **`comun.py`** — kernels y utilidades compartidas; **`graficar_gpu.py`** — gráfico; **`colab_hito2.ipynb`** — notebook que reproduce todo en Colab.

---

## 2. Técnica de optimización: memoria compartida + reducción en árbol

El kernel optimizado (`reduce_shared` en `kernels.py`) tiene tres pasos, cada uno ligado a un concepto de clase:

1. **Carga coalescida a memoria compartida (Clase 6).** El hilo `i` lee `x[i]`: hilos consecutivos de un warp leen direcciones consecutivas, así que el hardware junta las lecturas en pocas transacciones. Los hilos de borde ($i \ge N$) aportan 0.
2. **Reducción en árbol (Clase 7).** En cada paso, la mitad de los hilos suma otra posición de `shared`: `stride` empieza en `hilos_por_bloque / 2` y se divide por 2 hasta llegar a 1, con `cuda.syncthreads()` entre pasos. Con 256 hilos son $\log_2 256 = 8$ pasos; en cada uno trabajan $128, 64, 32, \dots$ hilos: es **paralelo de verdad** (un `for` serial en el hilo 0 no lo sería).
3. **Un parcial por bloque.** El hilo 0 escribe el total del bloque en `parcial[bloque]`; el host suma los parciales (uno cada 256 elementos) en `float64`.

Con esto, los accesos a memoria global dejan de competir por una única dirección y pasan a ser **lecturas independientes y coalescidas**; la serialización queda limitada a los pasos del árbol dentro de un bloque.

---

## 3. Corrección y precisión

La validación es la del Hito 1 (`validar_suma`): la suma debe ser $N$ con tolerancia $10^{-6} \cdot N$. Dentro de la GPU se acumula en `float32` (el tipo de los kernels) y en el host en `float64`.

**Resultado medido con $N = 2^{25}$ (33 554 432):** el kernel ingenuo devolvió `16 777 216` (esperado `33 554 432`, `Checksum Válido: NO`), mientras que el optimizado devolvió `33 554 432` (`Checksum Válido: SÍ`). Es consecuencia de la precisión de `float32`: los enteros solo son exactos hasta $2^{24} = 16\,777\,216$, y sumar $1.0$ a un acumulador que ya vale $2^{24}$ no lo modifica; por eso el acumulador único del ingenuo **se satura**. El optimizado no sufre el problema porque cada bloque suma solo 256 elementos en `float32` y el total se junta en `float64`. Con $N \le 2^{24}$ (los tamaños del benchmark) ambos kernels dan el resultado exacto.

---

## 4. Metodología de medición

- **Tiempo del kernel con `cuda.event`**, no con el reloj de CPU: los lanzamientos de kernels son asincrónicos y `perf_counter` mediría solo el lanzamiento.
- **Calentamiento descartado:** la primera llamada de Numba **compila (JIT)** el kernel; se descarta una corrida antes de medir.
- **Mediana** de 9 repeticiones, igual que el Hito 1. Para el ingenuo, el acumulador se pone en cero **antes** de iniciar el cronómetro.
- **Transferencias aparte:** `H2D` es la copia host→device del vector. Se reporta por separado porque una GPU rápida puede no convenir si hay que copiar los datos.
- **Ancho de banda efectivo:** $GB/s = \dfrac{4 \cdot N\ \text{bytes}}{t}$ (cada elemento `float32` se lee al menos una vez).
- **Referencia de CPU:** el mismo `np.sum(x, dtype=np.float64)` del `secuencial.py` del Hito 1, **medido en la misma sesión** de Colab.
- Las columnas *opt/ing* y *opt/CPU* son cocientes de tiempos de **kernel** (sin copia).

---

## 5. Resultados

**Hardware y entorno (2026-10-21, Google Colab):** GPU Tesla T4 (compute capability 7.5, 15 360 MiB), driver 580.82.07, CUDA 13.0; Python 3.13; Numba 0.61.2 (soporte CUDA de `numba_cuda`). Repeticiones: 9 (mediana, calentamiento descartado), 256 hilos por bloque.  
**Comando:** `python benchmark_gpu.py --reps 9`

| $N$ | CPU (ms) | GPU ingenua (ms) | GPU optimizada (ms) | opt / ingenua | opt / CPU (kernel) | GB/s ingenua | GB/s optimizada | H2D (ms) | Checksum |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| $2^{20}$ | 0.640 | 3.608 | 0.111 | 32.6× | 5.78× | 1.2 | 37.8 | 1.315 | OK/OK |
| $2^{22}$ | 2.680 | 14.274 | 0.287 | 49.8× | 9.35× | 1.2 | 58.6 | 3.741 | OK/OK |
| $2^{24}$ | 12.056 | 33.282 | 0.624 | 53.4× | 19.34× | 2.0 | 107.6 | 15.320 | OK/OK |

**Barrido de hilos por bloque** (kernel optimizado, $N = 2^{24}$):

| Hilos/bloque | Bloques | Kernel (ms) | GB/s | Checksum |
| :---: | :---: | :---: | :---: | :---: |
| 128 | 131 072 | 0.598 | 112.2 | OK |
| 256 | 65 536 | 0.639 | 105.0 | OK |
| 512 | 32 768 | 0.703 | 95.5 | OK |

Otras corridas de referencia ($N = 2^{24}$, scripts por separado, mediana de 7): ingenuo 36.147 ms, optimizado 1.342 ms. El optimizado tardó más que en el benchmark (0.624 ms) porque el script suelto arranca con la GPU "en frío"; para comparar se usan los valores del benchmark, que incluyen calentamiento. Hay variación entre corridas (el mismo caso de 256 hilos dio 0.624 ms y 0.639 ms dentro de la misma ejecución).

Salidas completas: `resultados_gpu.txt`, `resultados_gpu.csv`, `resultados_threads.csv`. Gráfico: `speedup_gpu.png`.

---

## 6. Perfilado con Nsight Compute

Comandos (Nsight Compute 2025.3.1, $N = 2^{24}$, una sola corrida del kernel):

```bash
ncu --set basic -k regex:suma_atomica  --launch-count 1 -o ingenuo    python kernel_ingenuo.py 16777216 1
ncu --set full  -k regex:reduce_shared --launch-count 1 -o optimizado python kernel_optimizado.py 16777216 1
ncu --import <reporte>.ncu-rep --page details
```

El primer intento con `--set full` sobre el kernel ingenuo se interrumpió antes de terminar, por lo que se volvió a correr con el conjunto `basic` (el kernel optimizado sí se perfiló con `full`). Reportes: `ingenuo.ncu-rep`, `optimizado.ncu-rep`.

| Métrica | Ingenua | Optimizada |
| :--- | :---: | :---: |
| Duración (bajo ncu) | 58.87 ms | 1.52 ms |
| Memory Throughput | 2.06 % | 83.51 % |
| DRAM Throughput | 0.51 % | 19.32 % (61.76 GB/s) |
| Compute (SM) Throughput | 1.22 % | 83.51 % |
| Ocupación teórica | 100 % | 100 % |
| Ocupación lograda | 87.63 % | 88.72 % |

Las duraciones bajo `ncu` son mayores que las medidas con `cuda.event` porque el perfilador fija los relojes de la GPU; por eso los tiempos de comparación son los de la sección 5 y estas métricas se usan para ver **dónde está el cuello de botella**. Con el 19.32 % de DRAM = 61.76 GB/s, el pico de memoria de esta T4 es de unos 320 GB/s.

---

## 7. Análisis

1. **El `atomicAdd` sobre una única dirección serializa.** El ingenuo es de 32× a 53× más lento que el optimizado, y la diferencia crece con $N$ (32.6× → 49.8× → 53.4×). Su ancho de banda efectivo queda en 1.2–2.0 GB/s, muy por debajo del pico (~320 GB/s). Con `ncu`, el kernel ingenuo usa solo el 2 % de la memoria y el 1.2 % del cómputo, **aunque su ocupación es alta (87.6 %)**: hay muchos warps residentes, pero están esperando su turno para actualizar la misma dirección. El problema no es falta de hilos sino la contención.
2. **El optimizado aprovecha mucho mejor la GPU.** Alcanza 107.6 GB/s con $N = 2^{24}$ (alrededor de un tercio del pico) y su ancho de banda sube con $N$ (37.8 → 58.6 → 107.6 GB/s): con tamaños chicos pesan más los costos fijos de lanzar el kernel. Según `ncu`, no está limitado por la DRAM (19.3 %) sino que sus unidades de memoria on-chip y de cómputo están al 83.5 %. **Hipótesis (no verificada con más métricas):** el cuello de botella es el trabajo dentro del bloque —escrituras y lecturas en memoria compartida y `syncthreads` del árbol—. Una mejora posible, que no se probó, es que cada hilo sume varios elementos en un registro antes de entrar al árbol.
3. **CPU vs. GPU.** Solo con el kernel, la GPU optimizada es 19.3× más rápida que la CPU con $N = 2^{24}$ (0.624 vs. 12.056 ms). Pero copiar el vector a la GPU tarda 15.320 ms: **más que sumarlo en la CPU**. El tiempo total del optimizado (kernel + copia de parciales + suma final en el host) es 0.819 ms; sumándole la copia del vector da ≈ 16.1 ms contra 12.1 ms de la CPU. Para una suma aislada no conviene mover los datos; la GPU se justifica cuando los datos ya están en ella o se hace mucho más cómputo por dato. (Una copia desde memoria *pinned* podría ser más rápida; no se probó.)
4. **Tamaño de bloque.** 128 hilos fue el más rápido (0.598 ms) y 512 el más lento (0.703 ms, ~18 % más que 128). Entre 128 y 256 la diferencia (6.8 %) es comparable al ruido observado entre corridas (~2.5 %), por lo que no es concluyente. **Hipótesis:** con más hilos el árbol tiene más pasos y más hilos ociosos; no se perfiló.
5. **Precisión en `float32`** (sección 3): se confirmó que para $N > 2^{24}$ el acumulador único del ingenuo se satura y da un resultado **incorrecto**, mientras que el optimizado sigue siendo exacto.
6. **Conexión con la teoría:** el modelo SIMT y los warps explican por qué muchos hilos esperando la misma dirección no suman rendimiento (Clase 5); la jerarquía de memoria, la memoria compartida y la coalescencia explican la mejora del optimizado (Clase 6); la reducción en árbol y el perfilado con Nsight Compute (Clase 7).

---

## 8. Instrucciones de ejecución

**En Google Colab (no hace falta placa NVIDIA propia):**
1. Abrir `colab_hito2.ipynb` (*Archivo → Abrir cuaderno → GitHub*, rama `laura-hito2`).
2. *Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU (T4)*.
3. Ejecutar las celdas en orden y descargar los resultados antes de cerrar la sesión.

**Sin GPU (solo para validar la lógica), con el simulador de Numba:**

```bash
pip install numba numpy matplotlib
NUMBA_ENABLE_CUDASIM=1 python kernel_optimizado.py 4096 2      # en PowerShell: $env:NUMBA_ENABLE_CUDASIM=1
```

El simulador corre cada hilo de CUDA en CPU: sirve para comprobar que el resultado es correcto; los tiempos no representan a la GPU.

**Con GPU NVIDIA local:**

```bash
python kernel_ingenuo.py 16777216
python kernel_optimizado.py 16777216
python benchmark_gpu.py --reps 9 | tee resultados_gpu.txt
python graficar_gpu.py resultados_gpu.csv speedup_gpu.png
```

---

## 9. Archivos

| Archivo | Rol |
| :--- | :--- |
| `kernels.py` | Kernels `suma_atomica` y `reduce_shared`, y la medición con `cuda.event`. |
| `kernel_ingenuo.py`, `kernel_optimizado.py` | Ejecutan cada versión y validan el resultado. |
| `benchmark_gpu.py` | Compara CPU vs. GPU ingenua vs. optimizada; barrido de hilos por bloque; escribe los CSV. |
| `graficar_gpu.py` | Genera `speedup_gpu.png` desde `resultados_gpu.csv`. |
| `comun.py` | Vector de unos, validación de la suma y descripción del entorno. |
| `colab_hito2.ipynb` | Reproduce todo en Colab. |
| `resultados_gpu.txt/.csv`, `resultados_threads.csv`, `speedup_gpu.png` | Salidas medidas en Colab. |
| `ingenuo.ncu-rep`, `optimizado.ncu-rep` | Reportes de Nsight Compute. |
| `informe-hito2.pdf/.docx/.md` | Informe de 2 páginas. |

---

## 10. Criterio de "listo"

- [x] Ambas versiones corren en GPU real (Tesla T4) con `Checksum Válido: SÍ` para $N \le 2^{24}$.
- [x] Técnica visible en el código optimizado (memoria compartida + árbol + carga coalescida).
- [x] Tabla de tiempos propia con GPU y versión de Numba declaradas (mediana de 9, calentamiento descartado).
- [x] Datos de perfilado con Nsight Compute para ambos kernels.
- [x] Análisis con los datos medidos y conexión con las Clases 5–7.
- [x] Informe de 2 páginas: CPU del Hito 1 vs. GPU ingenua vs. GPU optimizada.
