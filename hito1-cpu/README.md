# Hito 1 — Concurrencia en CPU

**Objetivo:** demostrar dominio de hilos y sincronización.

## Enunciado

Problema con paralelismo de datos claro: multiplicación de matrices C = A × B
(N×N, cuadradas). Versión secuencial (`secuencial.cpp`) y versión multi-hilo
(`paralelo.cpp`), con una primitiva de sincronización de la Clase 3
(`std::mutex`, protegiendo un reparto dinámico de filas entre los hilos).
Medir aceleración S(T) = T_secuencial / T(T) vs. número de hilos y comparar
con la Ley de Amdahl.

## Archivos

| Archivo | Qué hay |
| --- | --- |
| `secuencial.cpp` | Multiplicación en un solo hilo. Referencia completa. |
| `paralelo.cpp` | Multiplicación multi-hilo, con `std::mutex` para el reparto dinámico de filas. |
| `benchmark.cpp` | Corre ambas versiones varias veces, calcula la mediana, el speedup medido, y lo compara contra la Ley de Amdahl. |
| `Makefile` | Compila los tres. |

## Compilación

```
make
./secuencial 1024
./paralelo 1024 4
./benchmark 1024 5 8
```

- `secuencial [N]`
- `paralelo [N] [num_hilos]`
- `benchmark [N] [repeticiones] [max_hilos]` — corre el barrido completo e
  imprime una tabla con tiempo, speedup medido y speedup teórico de Amdahl.

Compilador C++17. `make` agrega `-pthread`.

## Cómo se estima la fracción secuencial (s)

En vez de declarar `s` a ojo, `benchmark.cpp` la despeja de la propia
medición con la mayor cantidad de hilos, usando la fórmula de Amdahl al
revés:

```
S = 1 / (s + (1-s)/p)   =>   s = (p - S) / (S · (p - 1))
```

Con ese `s` calcula la curva teórica completa para todas las cantidades de
hilos y las muestra junto a la medida, en la misma tabla.

## Criterio de "listo"

- [x] `secuencial` y `paralelo` dan el mismo checksum (`benchmark` lo verifica
      automáticamente en cada corrida).
- [x] `paralelo` usa una primitiva de sincronización real (`std::mutex`
      protegiendo el contador de filas pendientes).
- [x] `benchmark` imprime T(hilos) y S(hilos) para varias cantidades de hilos.
- [ ] Informe de 1–2 páginas con la curva medida vs. teórica y el análisis de
      por qué difieren (memoria, join, sobre-suscripción, tamaño de N, etc.) —
      corriendo `./benchmark` en una máquina con varios núcleos reales.

## Entregable

Código (ya compila y corre) + informe corto (1–2 páginas) con la tabla/curva
de `benchmark` y una explicación de las diferencias entre lo medido y lo
predicho por Amdahl.