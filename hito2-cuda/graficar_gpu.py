"""Genera speedup_gpu.png (tiempos y ancho de banda) desde resultados_gpu.csv.
Uso: python graficar_gpu.py [resultados_gpu.csv] [salida.png]"""
import csv
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

entrada = sys.argv[1] if len(sys.argv) > 1 else "resultados_gpu.csv"
salida = sys.argv[2] if len(sys.argv) > 2 else "speedup_gpu.png"
filas = list(csv.DictReader(open(entrada, encoding="utf-8")))
n = [int(f["n"]) for f in filas]
col = lambda k: [float(f[k]) for f in filas]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
a1.plot(n, col("cpu_ms"), "o-", label="CPU secuencial (Hito 1)")
a1.plot(n, col("ingenuo_ms"), "s-", label="GPU ingenua (atomicAdd)")
a1.plot(n, col("optimizado_ms"), "^-", label="GPU optimizada (shared + árbol)")
a1.set_xscale("log", base=2); a1.set_yscale("log")
a1.set_xlabel("N (elementos)"); a1.set_ylabel("Tiempo (ms, mediana)")
a1.set_title("Tiempo de la reducción"); a1.grid(True, which="both", alpha=.3); a1.legend()

a2.plot(n, col("gbps_ingenuo"), "s-", label="GPU ingenua")
a2.plot(n, col("gbps_optimizado"), "^-", label="GPU optimizada")
a2.set_xscale("log", base=2)
a2.set_xlabel("N (elementos)"); a2.set_ylabel("GB/s efectivos (4·N bytes / tiempo)")
a2.set_title("Ancho de banda efectivo"); a2.grid(True, alpha=.3); a2.legend()
fig.savefig(salida, dpi=150, bbox_inches="tight")
print("guardado", salida)
