# Makefile for the CUDA sources accompanying
# "Toward Optimizing Networking and Cybersecurity Applications Using
#  Domain-Specific Accelerators for Dynamic Programming".
#
# App1 is the network resilience system, all-pairs shortest path by
# Floyd-Warshall. App2 is the deep packet inspection system, signature matching
# by Smith-Waterman. Binaries are built next to their sources, and their names
# carry a tag naming the GPU they were built for, so one shared file system can
# hold a build per machine and both can be measured at the same time.
#
# On a cluster that provides its toolchain through environment modules, load the
# compiler first:
#
#   module unload gcc
#   module load gcc/12.2.0
#   module load cuda12.4/
#
# Then, on the A100 machine and the H200 machine respectively:
#
#   make ARCHES=80 TAG=a100
#   make ARCHES=90 TAG=h200
#
# which produce App1/floyd_warshall_routing-a100 and
# App1/floyd_warshall_routing-h200. Other targets:
#
#   make                          build both applications, both architectures
#   make app1                     build the network resilience system
#   make app2                     build the deep packet inspection system
#   make check                    run the eight routing variants and check them
#   make clean                    remove the binaries, every tag
#   make help                     print the variables and their current values

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

# Name tag appended to every binary. Two machines that share a file system would
# otherwise overwrite each other's binaries, and a build for the wrong GPU fails
# at run time with "no kernel image is available for execution on the device".
# The default names the architectures that were built; override it with a
# machine name, for example TAG=h200, when that reads better in a log.
empty :=
space := $(empty) $(empty)
TAG   ?= $(subst $(space),-,$(addprefix sm,$(ARCHES)))

APP1 := App1
APP2 := App2

# Programs that sample GPU power, so they link NVML and pthreads.
NVML_BASE := \
	$(APP1)/floyd_warshall_routing \
	$(APP2)/dpi_memory_focused \
	$(APP2)/dpi_memory_focused_energy

# Programs that need neither library.
PLAIN_BASE := \
	$(APP2)/dpi_occupancy_focused \
	$(APP2)/dpi_occupancy_focused_variant \
	$(APP2)/dpi_regex_matching

BASE_TARGETS  := $(NVML_BASE) $(PLAIN_BASE)
NVML_TARGETS  := $(addsuffix -$(TAG),$(NVML_BASE))
PLAIN_TARGETS := $(addsuffix -$(TAG),$(PLAIN_BASE))
APP1_TARGETS  := $(APP1)/floyd_warshall_routing-$(TAG)
APP2_TARGETS  := $(filter $(APP2)/%,$(NVML_TARGETS) $(PLAIN_TARGETS))
TARGETS       := $(NVML_TARGETS) $(PLAIN_TARGETS)

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
$(NVML_TARGETS): %-$(TAG): %.cu Makefile
	$(NVCC) $(NVCCFLAGS) $< $(NVML_LIBS) -o $@

$(PLAIN_TARGETS): %-$(TAG): %.cu Makefile
	$(NVCC) $(NVCCFLAGS) $< -o $@

# Runs all eight routing kernel variants once and prints one line each. The last
# field is the number of matrix entries that disagree with the closed-form
# answer, and it must be 0 on every line.
check: $(APP1_TARGETS)
	@for l in coalesced strided; do \
	  for d in on off; do \
	    for s in always changed; do \
	      printf '%-10s dpx=%-3s store=%-7s ' "$$l" "$$d" "$$s"; \
	      ./$(APP1_TARGETS) --nodes $(CHECK_NODES) --trials 1 \
	        --warmup 0 --layout $$l --dpx $$d --store $$s \
	        | grep -v '^#' | tail -1; \
	    done; \
	  done; \
	done

# Removes every tag, not only the current one, and the untagged binaries left by
# earlier versions of this file.
clean:
	rm -f $(foreach t,$(BASE_TARGETS),$(t) $(t)-*)

help:
	@echo "Targets:"
	@echo "  all      build both applications (default)"
	@echo "  app1     build the network resilience system only"
	@echo "  app2     build the deep packet inspection system only"
	@echo "  check    build App1 and run its eight kernel variants"
	@echo "  clean    remove the built binaries, every tag"
	@echo ""
	@echo "Variables and their current values:"
	@echo "  NVCC        = $(NVCC)"
	@echo "  ARCHES      = $(ARCHES)   (80 is the A100, 90 the H100 and H200)"
	@echo "  TAG         = $(TAG)"
	@echo "  NVCCFLAGS   = $(NVCCFLAGS)"
	@echo "  NVML_LIBS   = $(NVML_LIBS)"
	@echo "  CHECK_NODES = $(CHECK_NODES)"
	@echo ""
	@echo "Binaries this invocation would build:"
	@echo "  $(TARGETS)"
