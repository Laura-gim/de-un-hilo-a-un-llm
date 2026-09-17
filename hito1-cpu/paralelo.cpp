// paralelo.cpp
// Hito 1 — Concurrencia en CPU
// Multiplicación de matrices C = A x B, version MULTI-HILO (std::thread).
//
// Cada hilo calcula filas completas de C. El reparto de filas es
// DINAMICO: hay un contador compartido ("proxima fila libre") protegido
// con std::mutex (primitiva de sincronizacion de la Clase 3). Cada hilo,
// apenas termina una fila, pide la siguiente — asi se balancea mejor la
// carga que con un reparto fijo.
//
// Compilar: g++ -O2 -std=c++17 -pthread -o paralelo paralelo.cpp
// Correr:   ./paralelo [N] [num_hilos]

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <thread>
#include <mutex>
#include <cstdlib>

using Matriz = std::vector<double>;

constexpr unsigned SEMILLA = 42; // misma semilla que secuencial.cpp

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

// --- Estado compartido para el reparto dinamico de filas ---
std::mutex mtxFila;
int siguienteFila = 0;

int obtenerSiguienteFila(int n) {
    std::lock_guard<std::mutex> lock(mtxFila); // seccion critica
    if (siguienteFila >= n) return -1;
    return siguienteFila++;
}

void trabajador(const Matriz &A, const Matriz &B, Matriz &C, int n) {
    while (true) {
        int fila = obtenerSiguienteFila(n);
        if (fila == -1) break;
        for (int k = 0; k < n; ++k) {
            double aik = A[fila * n + k];
            for (int j = 0; j < n; ++j) {
                C[fila * n + j] += aik * B[k * n + j];
            }
        }
    }
}

Matriz multiplicar(const Matriz &A, const Matriz &B, int n, int numHilos) {
    Matriz C(static_cast<size_t>(n) * n, 0.0);
    siguienteFila = 0; // reiniciar el contador para esta corrida

    std::vector<std::thread> hilos;
    for (int t = 0; t < numHilos; ++t) {
        hilos.emplace_back(trabajador, std::cref(A), std::cref(B), std::ref(C), n);
    }
    for (auto &h : hilos) h.join();
    return C;
}

int main(int argc, char *argv[]) {
    int n = 512;
    int numHilos = static_cast<int>(std::thread::hardware_concurrency());
    if (numHilos <= 0) numHilos = 4;

    if (argc > 1) n = std::atoi(argv[1]);
    if (argc > 2) numHilos = std::atoi(argv[2]);
    if (n <= 0) { std::cerr << "N invalido, uso 512.\n"; n = 512; }
    if (numHilos <= 0) { std::cerr << "num_hilos invalido, uso 4.\n"; numHilos = 4; }

    std::cout << "Multiplicacion paralela, N = " << n << ", hilos = " << numHilos << "\n";

    Matriz A = generarMatriz(n, SEMILLA);
    Matriz B = generarMatriz(n, SEMILLA + 1);

    auto t0 = std::chrono::high_resolution_clock::now();
    Matriz C = multiplicar(A, B, n, numHilos);
    auto t1 = std::chrono::high_resolution_clock::now();
    double segundos = std::chrono::duration<double>(t1 - t0).count();

    std::cout << "Tiempo: " << segundos << " s\n";
    std::cout << "Checksum: " << checksum(C) << "\n";

    return 0;
}