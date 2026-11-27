# Сборка и запуск: make full
CXX = mpicxx
CXXFLAGS = -O2 -std=c++17 -Wall
NP ?= 4
# Если процессов больше, чем ядер, OpenMPI требует --oversubscribe; MPICH его не знает.
OVERSUB := $(shell mpirun --version 2>&1 | grep -qi "open mpi\|open-mpi" && echo --oversubscribe)

all: mpi_collective

mpi_collective: main.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

full: mpi_collective
	mpirun $(OVERSUB) -np $(NP) ./mpi_collective

clean:
	rm -f mpi_collective

.PHONY: all full clean
