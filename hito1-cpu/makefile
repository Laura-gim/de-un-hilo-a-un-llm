CXX = g++
CXXFLAGS = -O2 -std=c++17 -Wall -Wextra
LDFLAGS = -pthread

.PHONY: all clean

all: secuencial paralelo benchmark

secuencial: secuencial.cpp
	$(CXX) $(CXXFLAGS) -o secuencial secuencial.cpp

paralelo: paralelo.cpp
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o paralelo paralelo.cpp

benchmark: benchmark.cpp
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o benchmark benchmark.cpp

clean:
	rm -f secuencial paralelo benchmark