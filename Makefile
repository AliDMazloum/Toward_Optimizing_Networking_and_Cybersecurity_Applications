# Makefile for the CUDA sources accompanying
# "Toward Optimizing Networking and Cybersecurity Applications Using
#  Domain-Specific Accelerators for Dynamic Programming".
#
# App1 is the network resilience system, all-pairs shortest path by
# Floyd-Warshall. App2 is the deep packet inspection system, signature matching
# by Smith-Waterman. Binaries are built next to their sources.
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
#   make                          build both applications
#   make app1                     build the network resilience system
#   make app2                     build the deep packet inspection system
#   make ARCH=sm_80               build for another GPU, for example the A100
#   make check                    run the four routing variants and check them
#   make clean                    remove the binaries

NVCC      ?= nvcc
ARCH      ?= sm_90
NVCCFLAGS ?= -O3 -arch=$(ARCH)
NVML_LIBS ?= -lnvidia-ml -lpthread

APP1 := App1
APP2 := App2

# Programs that sample GPU power, so they link NVML and pthreads.
NVML_TARGETS := \
	$(APP1)/floyd_warshall_routing \
	$(APP2)/dpi_memory_focused \
	$(APP2)/dpi_memory_focused_energy

# Programs that need neither library.
PLAIN_TARGETS := \
	$(APP2)/dpi_occupancy_focused \
	$(APP2)/dpi_occupancy_focused_variant \
	$(APP2)/dpi_regex_matching

APP1_TARGETS := $(APP1)/floyd_warshall_routing
APP2_TARGETS := $(filter $(APP2)/%,$(NVML_TARGETS) $(PLAIN_TARGETS))
TARGETS      := $(NVML_TARGETS) $(PLAIN_TARGETS)

# Problem size used by the check target. Small enough to finish quickly.
CHECK_NODES ?= 1000

.PHONY: all app1 app2 check clean help

all: $(TARGETS)

app1: $(APP1_TARGETS)

app2: $(APP2_TARGETS)

$(NVML_TARGETS): %: %.cu
	$(NVCC) $(NVCCFLAGS) $< $(NVML_LIBS) -o $@

$(PLAIN_TARGETS): %: %.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

# Runs each of the four routing kernel variants once. The last field of every
# trial line is the number of matrix entries that disagree with the closed-form
# answer, and it must be 0 in all four runs.
check: $(APP1_TARGETS)
	./$(APP1)/floyd_warshall_routing --nodes $(CHECK_NODES) --trials 1 --warmup 0 --layout coalesced --dpx on
	./$(APP1)/floyd_warshall_routing --nodes $(CHECK_NODES) --trials 1 --warmup 0 --layout coalesced --dpx off
	./$(APP1)/floyd_warshall_routing --nodes $(CHECK_NODES) --trials 1 --warmup 0 --layout strided --dpx on
	./$(APP1)/floyd_warshall_routing --nodes $(CHECK_NODES) --trials 1 --warmup 0 --layout strided --dpx off

clean:
	rm -f $(TARGETS)

help:
	@echo "Targets:"
	@echo "  all      build both applications (default)"
	@echo "  app1     build the network resilience system only"
	@echo "  app2     build the deep packet inspection system only"
	@echo "  check    build App1 and run its four kernel variants"
	@echo "  clean    remove the built binaries"
	@echo ""
	@echo "Variables and their current values:"
	@echo "  NVCC        = $(NVCC)"
	@echo "  ARCH        = $(ARCH)   (sm_90 for H100 and H200, sm_80 for A100)"
	@echo "  NVCCFLAGS   = $(NVCCFLAGS)"
	@echo "  NVML_LIBS   = $(NVML_LIBS)"
	@echo "  CHECK_NODES = $(CHECK_NODES)"
