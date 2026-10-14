// benchmark.cpp
// Hito 1 — Concurrencia en CPU
// Corre la version secuencial y la paralela (con distintas cantidades de
// hilos), varias veces cada una, y muestra una tabla con tiempo mediano,
// speedup medido y speedup teorico segun la Ley de Amdahl.
//
// Compilar: g++ -O2 -std=c++17 -pthread -o benchmark benchmark.cpp
// Correr:   ./benchmark [N] [repeticiones] [max_hilos]

#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <chrono>
#include <thread>
#include <mutex>
#include <algorithm>
#include <cstdlib>

using Matriz = std::vector<double>;

constexpr unsigned SEMILLA = 42;

Matriz generarMatriz(int n, unsigned semilla) {
    Matriz m(static_cast<size_t>(n) * n);
    std::mt19937 gen(semilla);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (auto &v : m) v = dist(gen);
    return m;
}

double checksum(const Matriz &C) {
    double s = 0.0;
    for (double v : C) s += v;
    return s;
}

Matriz multiplicarSecuencial(const Matriz &A, const Matriz &B, int n) {
    Matriz C(static_cast<size_t>(n) * n, 0.0);
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < n; ++k) {
            double aik = A[i * n + k];
            for (int j = 0; j < n; ++j) C[i * n + j] += aik * B[k * n + j];
        }
    }
    return C;
}

std::mutex mtxFila;
int siguienteFila = 0;

int obtenerSiguienteFila(int n) {
    std::lock_guard<std::mutex> lock(mtxFila);
    if (siguienteFila >= n) return -1;
    return siguienteFila++;
}

void trabajador(const Matriz &A, const Matriz &B, Matriz &C, int n) {
    while (true) {
        int fila = obtenerSiguienteFila(n);
        if (fila == -1) break;
        for (int k = 0; k < n; ++k) {
            double aik = A[fila * n + k];
            for (int j = 0; j < n; ++j) C[fila * n + j] += aik * B[k * n + j];
        }
    }
}

Matriz multiplicarParalelo(const Matriz &A, const Matriz &B, int n, int numHilos) {
    Matriz C(static_cast<size_t>(n) * n, 0.0);
    siguienteFila = 0;
    std::vector<std::thread> hilos;
    for (int t = 0; t < numHilos; ++t) {
        hilos.emplace_back(trabajador, std::cref(A), std::cref(B), std::ref(C), n);
    }
    for (auto &h : hilos) h.join();
    return C;
}

double mediana(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    if (n % 2 == 1) return v[n / 2];
    return (v[n / 2 - 1] + v[n / 2]) / 2.0;
}

// Con UNA medicion (speedup S con p hilos) se puede despejar la fraccion
// secuencial s de la formula de Amdahl S = 1 / (s + (1-s)/p):
//   s = (p - S) / (S * (p - 1))
// Se usa la medicion con mas hilos (la mas exigente) para estimar s, y
// con ese s se calcula la curva teorica completa para comparar.
double estimarS(double p, double S) {
    if (p <= 1.0 || S <= 0.0) return 0.0;
    double s = (p - S) / (S * (p - 1.0));
    if (s < 0.0) s = 0.0;
    if (s > 1.0) s = 1.0;
    return s;
}

double amdahl(double p, double s) {
    return 1.0 / (s + (1.0 - s) / p);
}

int main(int argc, char *argv[]) {
    int n = 1024;
    int repeticiones = 5;
    int maxHilos = static_cast<int>(std::thread::hardware_concurrency());
    if (maxHilos <= 0) maxHilos = 8;

    if (argc > 1) n = std::atoi(argv[1]);
    if (argc > 2) repeticiones = std::atoi(argv[2]);
    if (argc > 3) maxHilos = std::atoi(argv[3]);
    if (n <= 0) n = 1024;
    if (repeticiones <= 0) repeticiones = 5;
    if (maxHilos <= 0) maxHilos = 8;

    std::cout << "Benchmark: N=" << n << "  repeticiones=" << repeticiones
              << "  max_hilos=" << maxHilos
              << "  hardware_concurrency()=" << std::thread::hardware_concurrency() << "\n\n";

    Matriz A = generarMatriz(n, SEMILLA);
    Matriz B = generarMatriz(n, SEMILLA + 1);

    // --- Secuencial: linea base y verificacion de checksum ---
    std::vector<double> tiemposSeq;
    double checksumSeq = 0.0;
    for (int r = 0; r < repeticiones; ++r) {
        auto t0 = std::chrono::high_resolution_clock::now();
        Matriz C = multiplicarSecuencial(A, B, n);
        auto t1 = std::chrono::high_resolution_clock::now();
        tiemposSeq.push_back(std::chrono::duration<double>(t1 - t0).count());
        checksumSeq = checksum(C);
    }
    double tSeq = mediana(tiemposSeq);

    // --- Barrido de hilos: 1, 2, 4, 8, ... hasta maxHilos ---
    std::vector<int> hilosAProbar;
    for (int t = 1; t <= maxHilos; t *= 2) hilosAProbar.push_back(t);
    if (hilosAProbar.empty() || hilosAProbar.back() != maxHilos) hilosAProbar.push_back(maxHilos);

    struct Fila { int hilos; double tiempo; double speedup; bool checksumOk; };
    std::vector<Fila> filas;
    bool todosOk = true;

    for (int hilos : hilosAProbar) {
        std::vector<double> tiempos;
        bool ok = true;
        for (int r = 0; r < repeticiones; ++r) {
            auto t0 = std::chrono::high_resolution_clock::now();
            Matriz C = multiplicarParalelo(A, B, n, hilos);
            auto t1 = std::chrono::high_resolution_clock::now();
            tiempos.push_back(std::chrono::duration<double>(t1 - t0).count());
            if (std::abs(checksum(C) - checksumSeq) > 1e-3) ok = false;
        }
        double t = mediana(tiempos);
        filas.push_back({hilos, t, tSeq / t, ok});
        if (!ok) todosOk = false;
    }

    // --- Estimar s de Amdahl usando la medicion con mas hilos ---
    double sEstimado = estimarS(static_cast<double>(filas.back().hilos), filas.back().speedup);

    // --- Reporte ---
    std::cout << "Verificacion de correccion: ";
    std::cout << (todosOk ? "OK, todos los checksums coinciden con la version secuencial.\n"
                           : "ATENCION, algun checksum no coincide. Revisar la sincronizacion.\n");
    std::cout << "T_secuencial (mediana de " << repeticiones << " corridas) = " << tSeq << " s\n";
    std::cout << "Fraccion secuencial estimada (s, a partir de la medicion con mas hilos) = "
              << std::fixed << std::setprecision(4) << sEstimado << "\n\n";

    std::cout << std::left
              << std::setw(8) << "hilos"
              << std::setw(16) << "tiempo_s"
              << std::setw(16) << "speedup_medido"
              << std::setw(18) << "speedup_amdahl"
              << "checksum\n";
    for (const auto &f : filas) {
        std::cout << std::left << std::setw(8) << f.hilos
                   << std::setw(16) << std::setprecision(6) << f.tiempo
                   << std::setw(16) << std::setprecision(3) << f.speedup
                   << std::setw(18) << std::setprecision(3) << amdahl(f.hilos, sEstimado)
                   << (f.checksumOk ? "OK" : "MISMATCH") << "\n";
    }

    std::cout << "\nCSV (para pegar en una planilla si hace falta):\n";
    std::cout << "hilos,tiempo_s,speedup_medido,speedup_amdahl\n";
    for (const auto &f : filas) {
        std::cout << f.hilos << "," << f.tiempo << "," << f.speedup << ","
                   << amdahl(f.hilos, sEstimado) << "\n";
    }

    return todosOk ? 0 : 1;
}