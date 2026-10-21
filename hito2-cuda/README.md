# Hito 2 — Kernels CUDA optimizados (Versión Python / Numba)

**Materia:** Programación Concurrente y Paralela  
**Integrantes:** Laura & Gabriel  
**Repositorio:** [de-un-hilo-a-un-llm](https://github.com/Laura-gim/de-un-hilo-a-un-llm.git)

> **Estado de esta entrega:** el código está completo y su **lógica se verificó en CPU** con un emulador de CUDA (N chicos, tamaños de bloque 32–256, N que no es múltiplo del bloque). **Todavía no se corrió en una GPU real**: las tablas de la sección 5 y el análisis de la sección 7 están marcados con `COMPLETAR` y se rellenan **solo con datos medidos en Colab**. No hay ningún número de GPU inventado en este archivo.

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
2. **Reducción en árbol (Clase 7).** En cada paso, la mitad de los hilos suma otra posición de `shared`: `stride` empieza en `hilos_por_bloque / 2` y se divide por 2 hasta llegar a 1, con `cuda.syncthreads()` entre pasos. Con 256 hilos son $\log_2 256 = 8$ pasos, en cada uno trabajan $128, 64, 32, \dots$ hilos: es **paralelo de verdad** (un `for` serial en el hilo 0 no lo sería).
3. **Un parcial por bloque.** El hilo 0 escribe el total del bloque en `parcial[bloque]`; el host suma los parciales (uno cada 256 elementos) en `float64`.

Con esto, los $N$ accesos a memoria global del baseline pasan de competir por una dirección a ser **lecturas independientes y coalescidas**, y la serialización queda reducida a los pasos del árbol dentro de un bloque.

---

## 3. Corrección y precisión

La validación es la del Hito 1 (`validar_suma`): la suma debe ser $N$ con tolerancia $10^{-6} \cdot N$. Se acumula en `float32` dentro de la GPU (el tipo que usan los kernels) y en `float64` en el host.

**Detalle de precisión que hay que verificar en Colab:** en `float32` los enteros solo son exactos hasta $2^{24} = 16\,777\,216$; sumar $1.0$ a un acumulador que ya vale $2^{24}$ se redondea y **no cambia su valor**. Por eso, para $N > 2^{24}$ se espera que el acumulador único del kernel ingenuo **sature** en $16\,777\,216$ y el resultado sea incorrecto, mientras que el optimizado (parciales de a 256 elementos, suma final en `float64`) siga correcto. Esta es una **hipótesis** que prueba la celda 7 del notebook (`python kernel_ingenuo.py 33554432`); lo que ocurra se documenta en la sección 7.

---

## 4. Metodología de medición

- **Tiempo del kernel con `cuda.event`**, no con el reloj de CPU: los lanzamientos de kernels son asincrónicos y `perf_counter` mediría solo el lanzamiento.
- **Calentamiento descartado:** la primera llamada de Numba **compila (JIT)** el kernel; se descarta una corrida antes de medir.
- **Mediana** de las repeticiones (7 por defecto), igual que el Hito 1. Para el ingenuo, el acumulador se pone en cero **antes** de iniciar el cronómetro.
- **Transferencias aparte:** `H2D` es la copia host→device del vector; `opt_total` incluye kernel + copia de parciales + suma final en el host. Se reportan por separado porque una GPU “rápida” puede perder frente a la CPU si hay que copiar los datos.
- **Ancho de banda efectivo:** $GB/s = \dfrac{4 \cdot N\ \text{bytes}}{t}$ (cada elemento `float32` se lee al menos una vez). Se compara con el pico de la GPU declarada.
- Referencia de CPU: el mismo `np.sum(x, dtype=np.float64)` del `secuencial.py` del Hito 1, **medido en la misma sesión** (el Hito 1 midió $T_{seq} \approx 13.850\ \text{ms}$ para $N = 2^{24}$ en Colab; se vuelve a medir para comparar en igualdad de condiciones).

---

## 5. Resultados (**COMPLETAR con la salida de `benchmark_gpu.py` en Colab**)

**Hardware:** GPU `COMPLETAR` (compute capability `COMPLETAR`), CPU de Colab `COMPLETAR`, Python `COMPLETAR`, Numba `COMPLETAR`. Repeticiones: `COMPLETAR` (mediana, calentamiento descartado).  
**Comando:** `python benchmark_gpu.py --reps 9`

| $N$ | CPU (ms) | GPU ingenua (ms) | GPU optimizada (ms) | opt / ingenua | opt / CPU (kernel) | GB/s ingenua | GB/s optimizada | H2D (ms) | Checksum |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| $2^{20}$ | | | | | | | | | |
| $2^{22}$ | | | | | | | | | |
| $2^{24}$ | | | | | | | | | |

**Barrido de hilos por bloque** (kernel optimizado, $N = 2^{24}$):

| Hilos/bloque | Bloques | Kernel (ms) | GB/s | Checksum |
| :---: | :---: | :---: | :---: | :---: |
| 128 | | | | |
| 256 | | | | |
| 512 | | | | |

Salidas completas: `resultados_gpu.txt`, `resultados_gpu.csv`, `resultados_threads.csv`. Gráfico: `speedup_gpu.png`.

---

## 6. Perfilado con Nsight Compute (**COMPLETAR o declarar la limitación**)

Se intenta perfilar **ambos** kernels (celda 8 del notebook):

```bash
ncu --set full -o ingenuo    python kernel_ingenuo.py 16777216
ncu --set full -o optimizado python kernel_optimizado.py 16777216
```

Numba compila en la primera llamada, por lo que `ncu` también verá la corrida de calentamiento: mirar los kernels `suma_atomica` y `reduce_shared`.

| Métrica | Ingenua | Optimizada |
| :--- | :---: | :---: |
| Tiempo del kernel (`gpu__time_duration`) | | |
| Ocupación (`sm__warps_active`) | | |
| Throughput de memoria (% del pico) | | |

Si `ncu` no está disponible o Colab no da permisos, se pega **el mensaje de error** acá y se usa como alternativa el tiempo de `cuda.event` y el ancho de banda efectivo calculado a mano (sección 4): **limitación declarada** (Clase 7).

---

## 7. Análisis (**COMPLETAR contrastando con los datos medidos**)

Las siguientes son las **hipótesis** que el informe debe confirmar o contradecir con los números de la sección 5; si los datos dicen otra cosa, se explica por qué.

1. **Serialización del `atomicAdd`.** Todos los hilos actualizan la misma dirección: se espera que el ingenuo sea el más lento y que su ancho de banda efectivo quede **muy por debajo** del pico de la GPU. `COMPLETAR`: ¿cuánto más lento fue (opt / ingenua)? ¿Crece o se mantiene al aumentar $N$?
2. **Reducción en árbol + memoria compartida.** Se espera que el optimizado se acerque al límite de **ancho de banda de memoria**: la suma es *memory-bound* (4 bytes leídos por cada suma), como ya se vio en el Hito 1 con la RAM. `COMPLETAR`: GB/s alcanzados frente al pico de la GPU declarada.
3. **CPU vs. GPU.** Comparar el tiempo del kernel contra la CPU, y también **incluyendo la copia H2D**: ¿la GPU gana si hay que copiar el vector? `COMPLETAR` con la columna `H2D` y `opt_total`.
4. **Tamaño de bloque.** ¿Cambia el rendimiento entre 128, 256 y 512 hilos por bloque? Relacionar con ocupación (warps activos por SM). `COMPLETAR`.
5. **Precisión `float32` para $N > 2^{24}$** (sección 3): ¿saturó el ingenuo? `COMPLETAR`.
6. **Conexión teórica:** modelo SIMT y warps (Clase 5), jerarquía de memoria y coalescencia (Clase 6), patrón de reducción (Clase 7).

---

## 8. Instrucciones de ejecución

**En Google Colab (recomendado, no hace falta placa NVIDIA propia):**
1. Abrir `colab_hito2.ipynb` (en Colab: *Archivo → Abrir cuaderno → GitHub* y pegar la URL del repo, o subir el archivo).
2. *Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU (T4)*.
3. Ejecutar las celdas en orden. Descargar los resultados antes de cerrar la sesión.

**Sin GPU (solo para validar la lógica), con el simulador de Numba:**

```bash
pip install numba numpy matplotlib
NUMBA_ENABLE_CUDASIM=1 python kernel_optimizado.py 4096 2      # en Windows PowerShell: $env:NUMBA_ENABLE_CUDASIM=1
NUMBA_ENABLE_CUDASIM=1 python benchmark_gpu.py --n 2048 4096 --reps 2 --threads 32 64
```

El simulador corre cada hilo de CUDA en CPU: sirve para comprobar que el resultado es correcto, **los tiempos no representan a la GPU**.

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
| `kernels.py` | Kernels `suma_atomica` y `reduce_shared`, y la función de medición con `cuda.event`. |
| `kernel_ingenuo.py`, `kernel_optimizado.py` | Ejecutan cada versión y validan el resultado. |
| `benchmark_gpu.py` | Compara CPU vs. GPU ingenua vs. optimizada; barrido de hilos por bloque; escribe los CSV. |
| `graficar_gpu.py` | Genera `speedup_gpu.png` desde `resultados_gpu.csv`. |
| `comun.py` | Vector de unos, validación de la suma y descripción del entorno (GPU y versión de Numba). |
| `colab_hito2.ipynb` | Reproduce todo en Colab, de punta a punta. |
| `informe-hito2-plantilla.md` | Plantilla del informe de 1–2 páginas. |

---

## 10. Criterio de "listo"

- [x] Ambas versiones implementadas y validadas con `validar_suma` en el emulador de CPU.
- [x] Técnica visible en el código optimizado (memoria compartida + árbol + carga coalescida).
- [ ] Corrida en GPU real: ambas versiones con `Checksum Válido: SÍ`.
- [ ] Tabla de tiempos propia (sección 5) con GPU y versión de Numba declaradas.
- [ ] Dato de perfilado (sección 6) o limitación declarada con el error pegado.
- [ ] Análisis (sección 7) escrito con los datos medidos.
- [ ] Informe de 1–2 páginas (CPU Hito 1 vs. GPU ingenua vs. GPU optimizada).
