// secuencial.cpp
// Hito 1 — Concurrencia en CPU
// Multiplicación de matrices C = A x B, version SECUENCIAL (un solo hilo).
//
// Compilar: g++ -O2 -std=c++17 -o secuencial secuencial.cpp
// Correr:   ./secuencial [N]

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cstdlib>

using Matriz = std::vector<double>; // matriz NxN guardada como vector plano

// Semilla fija: A y B se generan siempre igual, tanto acá como en
// paralelo.cpp, para poder comparar los resultados de las dos versiones.
constexpr unsigned SEMILLA = 42;

Matriz generarMatriz(int n, unsigned semilla) {
    Matriz m(static_cast<size_t>(n) * n);
    std::mt19937 gen(semilla);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (auto &v : m) v = dist(gen);
    return m;
}

// Suma de todos los elementos: sirve como "huella digital" rápida para
// comparar el resultado contra la versión paralela sin imprimir la
// matriz entera.
double checksum(const Matriz &C) {
    double s = 0.0;
    for (double v : C) s += v;
    return s;
}

Matriz multiplicar(const Matriz &A, const Matriz &B, int n) {
    Matriz C(static_cast<size_t>(n) * n, 0.0);
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < n; ++k) {
            double aik = A[i * n + k];
            for (int j = 0; j < n; ++j) {
                C[i * n + j] += aik * B[k * n + j];
            }
        }
    }
    return C;
}

int main(int argc, char *argv[]) {
    int n = 512;
    if (argc > 1) n = std::atoi(argv[1]);
    if (n <= 0) { std::cerr << "N invalido, uso 512.\n"; n = 512; }

    std::cout << "Multiplicacion secuencial, N = " << n << "\n";

    Matriz A = generarMatriz(n, SEMILLA);
    Matriz B = generarMatriz(n, SEMILLA + 1);

    auto t0 = std::chrono::high_resolution_clock::now();
    Matriz C = multiplicar(A, B, n);
    auto t1 = std::chrono::high_resolution_clock::now();
    double segundos = std::chrono::duration<double>(t1 - t0).count();

    std::cout << "Tiempo: " << segundos << " s\n";
    std::cout << "Checksum: " << checksum(C) << "\n";

    return 0;
}