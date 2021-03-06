BINARIES = benchmark check depth_map
CC = gcc
CFLAGS = -O3 -DNDEBUG -g0 -std=c99 -Wall -march=haswell -fopenmp -pthread
LFLAGS = -lm -lpthread
OMP_NESTED = TRUE

default: clean check benchmark

debug: clean all_debug

all: $(BINARIES)

all_debug: CC += -DDEBUG -ggdb3
all_debug: all

benchmark: benchmark.o calc_depth_naive.o calc_depth_optimized.o utils.o
	$(CC) $(CFLAGS) -o $@ $^ $(LFLAGS)

check: calc_depth_naive.o calc_depth_optimized.o check.o utils.o
	$(CC) $(CFLAGS) -o $@ $^ $(LFLAGS)

depth_map: calc_depth_naive.o calc_depth_optimized.o depth_map.o utils.o
	$(CC) $(CFLAGS) -o $@ $^ $(LFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $*.c

clean:
	rm -rf *.o
	rm -rf $(BINARIES)
