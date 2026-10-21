"""Hito 2 - Benchmark: CPU secuencial (Hito 1) vs. GPU ingenua vs. GPU optimizada.

Uso:
  python benchmark_gpu.py                       # N = 2^20, 2^22, 2^24; 7 repeticiones
  python benchmark_gpu.py --n 1048576 16777216 --reps 9 --threads 128 256 512
Escribe resultados_gpu.csv (tabla principal) y resultados_threads.csv (barrido de hilos por bloque).

Todos los tiempos son la MEDIANA de las repeticiones, con una corrida de calentamiento descartada.
Los tiempos "kernel" se miden con cuda.event (sin copias); "H2D" es la copia host->device del vector;
"opt_total" incluye kernel + copia de parciales + suma final en el host.
"""
import argparse
import csv
import statistics
import sys
import time

import numpy as np
from numba import cuda

from comun import SIMULADOR, crear_vector, validar_suma, describir_entorno
from kernels import crear_reduccion_shared, cantidad_bloques, medir_ms, THREADS_POR_DEFECTO
from kernel_ingenuo import sumar_ingenuo, suma_atomica, poner_cero
from kernel_optimizado import sumar_optimizado


def mediana_ms(f, reps):
    f()                                                # calentamiento descartado
    t = []
    for _ in range(reps):
        t0 = time.perf_counter()
        f()
        t.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(t)


def gbps(n, ms):
    """Ancho de banda efectivo: bytes minimos leidos (4 por elemento float32) / tiempo."""
    return (4.0 * n) / (ms * 1e-3) / 1e9 if ms > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="+", default=[1 << 20, 1 << 22, 1 << 24])
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--threads", type=int, nargs="+", default=[128, 256, 512])
    ap.add_argument("--out", default="resultados_gpu.csv")
    ap.add_argument("--out-threads", default="resultados_threads.csv")
    a = ap.parse_args()

    print("=" * 78)
    print(" HITO 2: Benchmark CPU vs. GPU ingenua vs. GPU optimizada")
    print(f" {describir_entorno()}")
    print(f" Repeticiones: {a.reps} (mediana, calentamiento descartado) | Hilos/bloque: {THREADS_POR_DEFECTO}")
    if SIMULADOR:
        print(" ATENCION: simulador de CPU. Sirve para validar la logica; los tiempos NO son de GPU.")
    print("=" * 78)

    kernel = crear_reduccion_shared(THREADS_POR_DEFECTO)
    filas, todo_ok = [], True
    cab = (f"{'N':>10} | {'CPU ms':>8} | {'ingenuo ms':>10} | {'optim. ms':>9} | {'opt/ing':>7} | "
           f"{'opt/CPU':>7} | {'GB/s ing':>8} | {'GB/s opt':>8} | {'H2D ms':>7} | check")
    print(cab)
    print("-" * len(cab))

    for n in a.n:
        x = crear_vector(n)
        cpu_ms = mediana_ms(lambda: float(np.sum(x, dtype=np.float64)), a.reps)       # = secuencial del Hito 1

        h2d_ms = mediana_ms(lambda: (cuda.to_device(x), cuda.synchronize()), a.reps)  # copia host->device
        d_x = cuda.to_device(x)
        d_out = cuda.to_device(np.zeros(1, dtype=np.float32))
        bloques = cantidad_bloques(n, THREADS_POR_DEFECTO)
        d_parcial = cuda.device_array(bloques, dtype=np.float32)

        ing_ms = medir_ms(lambda: suma_atomica[bloques, THREADS_POR_DEFECTO](d_x, d_out, n),
                          preparar=lambda: poner_cero[1, 1](d_out), reps=a.reps)
        ok_ing = validar_suma(sumar_ingenuo(d_x, d_out, n), n)[0]

        opt_ms = medir_ms(lambda: kernel[bloques, THREADS_POR_DEFECTO](d_x, d_parcial, n), reps=a.reps)
        ok_opt = validar_suma(sumar_optimizado(kernel, d_x, d_parcial, n, THREADS_POR_DEFECTO), n)[0]
        opt_total_ms = mediana_ms(lambda: sumar_optimizado(kernel, d_x, d_parcial, n, THREADS_POR_DEFECTO), a.reps)

        todo_ok &= ok_opt and ok_ing
        fila = {
            "n": n, "cpu_ms": cpu_ms, "ingenuo_ms": ing_ms, "optimizado_ms": opt_ms,
            "speedup_opt_vs_ingenuo": ing_ms / opt_ms, "speedup_opt_vs_cpu": cpu_ms / opt_ms,
            "gbps_ingenuo": gbps(n, ing_ms), "gbps_optimizado": gbps(n, opt_ms),
            "h2d_ms": h2d_ms, "opt_total_ms": opt_total_ms,
            "ok_ingenuo": ok_ing, "ok_optimizado": ok_opt,
        }
        filas.append(fila)
        chk = ("OK" if ok_ing else "FALLA ing") + "/" + ("OK" if ok_opt else "FALLA opt")
        print(f"{n:>10} | {cpu_ms:>8.3f} | {ing_ms:>10.3f} | {opt_ms:>9.3f} | {fila['speedup_opt_vs_ingenuo']:>6.1f}x | "
              f"{fila['speedup_opt_vs_cpu']:>6.2f}x | {fila['gbps_ingenuo']:>8.1f} | {fila['gbps_optimizado']:>8.1f} | "
              f"{h2d_ms:>7.3f} | {chk}")

    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in filas)
    print(f"\n-> escrito {a.out}")
    print("   opt_total_ms = kernel + copia de parciales + suma final en el host (sin H2D).")

    # ------------------------------------------------ barrido de hilos por bloque (N mayor)
    n = max(a.n)
    x = crear_vector(n)
    d_x = cuda.to_device(x)
    print(f"\nBarrido de hilos por bloque (kernel optimizado, N = {n})")
    print(f"{'hilos/bloque':>12} | {'bloques':>8} | {'kernel ms':>9} | {'GB/s':>7} | check")
    barrido = []
    for t in a.threads:
        k = crear_reduccion_shared(t)
        b = cantidad_bloques(n, t)
        d_p = cuda.device_array(b, dtype=np.float32)
        ms = medir_ms(lambda: k[b, t](d_x, d_p, n), reps=a.reps)
        ok = validar_suma(sumar_optimizado(k, d_x, d_p, n, t), n)[0]
        todo_ok &= ok
        barrido.append({"n": n, "hilos_por_bloque": t, "bloques": b, "kernel_ms": round(ms, 4),
                        "gbps": round(gbps(n, ms), 2), "ok": ok})
        print(f"{t:>12} | {b:>8} | {ms:>9.3f} | {gbps(n, ms):>7.1f} | {'OK' if ok else 'FALLA'}")
    with open(a.out_threads, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(barrido[0].keys()))
        w.writeheader()
        w.writerows(barrido)
    print(f"-> escrito {a.out_threads}")
    print("\nBenchmark completado." if todo_ok else "\nATENCION: alguna verificacion fallo.")
    return 0 if todo_ok else 1


if __name__ == "__main__":
    sys.exit(main())
