# Makefile for the CUDA sources accompanying
# "Toward Optimizing Networking and Cybersecurity Applications Using
#  Domain-Specific Accelerators for Dynamic Programming".
#
# On a cluster that provides its toolchain through environment modules, load the
# compiler first:
#
#   module unload gcc
#   module load gcc/12.2.0
#   module load cuda12.4/
#
# Then:
#
#   make                          build every program
#   make floyd_warshall_routing   build one program
#   make ARCH=sm_80               build for another GPU, for example the A100
#   make check                    run the four routing variants and check them
#   make clean                    remove the binaries

NVCC      ?= nvcc
ARCH      ?= sm_90
NVCCFLAGS ?= -O3 -arch=$(ARCH)
NVML_LIBS ?= -lnvidia-ml -lpthread

# Programs that sample GPU power, so they link NVML and pthreads.
NVML_TARGETS  := dpi_memory_focused dpi_memory_focused_energy floyd_warshall_routing

# Programs that need neither library.
PLAIN_TARGETS := dpi_occupancy_focused dpi_occupancy_focused_variant dpi_regex_matching

TARGETS := $(NVML_TARGETS) $(PLAIN_TARGETS)

# Problem size used by the check target. Small enough to finish quickly.
CHECK_NODES ?= 1000

.PHONY: all check clean help

all: $(TARGETS)

$(NVML_TARGETS): %: %.cu
	$(NVCC) $(NVCCFLAGS) $< $(NVML_LIBS) -o $@

$(PLAIN_TARGETS): %: %.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

# Runs each of the four routing kernel variants once. The last field of every
# trial line is the number of matrix entries that disagree with the closed-form
# answer, and it must be 0 in all four runs.
check: floyd_warshall_routing
	./floyd_warshall_routing --nodes $(CHECK_NODES) --trials 1 --warmup 0 --layout coalesced --dpx on
	./floyd_warshall_routing --nodes $(CHECK_NODES) --trials 1 --warmup 0 --layout coalesced --dpx off
	./floyd_warshall_routing --nodes $(CHECK_NODES) --trials 1 --warmup 0 --layout strided --dpx on
	./floyd_warshall_routing --nodes $(CHECK_NODES) --trials 1 --warmup 0 --layout strided --dpx off

clean:
	rm -f $(TARGETS)

help:
	@echo "Targets:"
	@echo "  all      build every program (default)"
	@echo "  check    build the routing program and run its four kernel variants"
	@echo "  clean    remove the built binaries"
	@echo ""
	@echo "Variables and their current values:"
	@echo "  NVCC        = $(NVCC)"
	@echo "  ARCH        = $(ARCH)   (sm_90 for H100, sm_80 for A100)"
	@echo "  NVCCFLAGS   = $(NVCCFLAGS)"
	@echo "  NVML_LIBS   = $(NVML_LIBS)"
	@echo "  CHECK_NODES = $(CHECK_NODES)"
