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
#   make ARCHES=90                build for one architecture only
#   make check                    run the four routing variants and check them
#   make clean                    remove the binaries

NVCC ?= nvcc

# Architectures to build for. The default produces one binary carrying native
# code for both the A100 (sm_80) and the H100 or H200 (sm_90), so the same build
# can be measured on either GPU. Building for a single architecture and then
# running on the other one still works, because the driver compiles the embedded
# PTX, but it does not measure the same machine code, which is a silent way to
# get a wrong number. Override with, for example, ARCHES=90.
ARCHES    ?= 80 90
GENCODE   := $(foreach a,$(ARCHES),-gencode arch=compute_$(a),code=sm_$(a))
NVCCFLAGS ?= -O3 $(GENCODE)
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

# Every binary depends on this file as well as on its source, so a change to
# ARCHES or to any other flag forces a rebuild. Without that, a binary built for
# one GPU is silently reused on another and fails with "no kernel image is
# available for execution on the device".
$(NVML_TARGETS): %: %.cu Makefile
	$(NVCC) $(NVCCFLAGS) $< $(NVML_LIBS) -o $@

$(PLAIN_TARGETS): %: %.cu Makefile
	$(NVCC) $(NVCCFLAGS) $< -o $@

# Runs all eight routing kernel variants once and prints one line each. The last
# field is the number of matrix entries that disagree with the closed-form
# answer, and it must be 0 on every line.
check: $(APP1_TARGETS)
	@for l in coalesced strided; do \
	  for d in on off; do \
	    for s in always changed; do \
	      printf '%-10s dpx=%-3s store=%-7s ' "$$l" "$$d" "$$s"; \
	      ./$(APP1)/floyd_warshall_routing --nodes $(CHECK_NODES) --trials 1 \
	        --warmup 0 --layout $$l --dpx $$d --store $$s \
	        | grep -v '^#' | tail -1; \
	    done; \
	  done; \
	done

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
	@echo "  ARCHES      = $(ARCHES)   (80 is the A100, 90 the H100 and H200)"
	@echo "  NVCCFLAGS   = $(NVCCFLAGS)"
	@echo "  NVML_LIBS   = $(NVML_LIBS)"
	@echo "  CHECK_NODES = $(CHECK_NODES)"
