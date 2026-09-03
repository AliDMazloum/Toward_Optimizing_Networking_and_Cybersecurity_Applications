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
# Then use the target named after the machine you are on. It fixes the
# architecture, the binary name and the output file, so the two machines cannot
# collide and there is nothing to remember at the prompt:
#
#   make a100                     build everything for the A100
#   make h200                     build everything for the H200
#   make a100-app1                the network resilience system only
#   make h200-sweep               build App1 and run the reported sweep
#   make a100-clean               remove this machine's binaries only
#
# Every one of those refuses to run a sweep if the node's GPU is not the one the
# target names, so a job landing on the wrong machine fails immediately instead
# of producing a mislabelled number.
#
# The general targets underneath take ARCHES and TAG by hand:
#
#   make                          build both applications, both architectures
#   make app1                     build the network resilience system
#   make app2                     build the deep packet inspection system
#   make check                    run the twelve routing variants and check them
#   make sweep                    run the reported App1 sweep for this TAG
#   make clean                    remove this tag's binaries
#   make clean-all                remove every tag, all machines
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

# The App1 sweep reported in the paper. Node counts are the topology sizes the
# manuscript uses; the flags are the protocol the SC'25 measurements followed, so
# the two are comparable. Each machine writes its own file, because two processes
# appending to one file on a shared mount interleave their rows.
SWEEP_NODES  ?= 1000 2000 3000 6000 12000 24000
SWEEP_TRIALS ?= 5
SWEEP_WARMUP ?= 1
SWEEP_FLAGS  ?= --layout coalesced --store changed --sync per-launch
SWEEP_CSV    ?= app1_final_$(TAG).csv

# Substring the node's GPU name must contain before a sweep runs. The machine
# targets set it; set it to nothing to skip the check.
GPU_MATCH ?=

.PHONY: all app1 app2 check sweep clean clean-all help
.PHONY: a100 a100-app1 a100-app2 a100-check a100-sweep a100-clean
.PHONY: h200 h200-app1 h200-app2 h200-check h200-sweep h200-clean

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

# Runs all twelve routing kernel variants once and prints one line each. The last
# field is the number of matrix entries that disagree with the closed-form
# answer, and it must be 0 on every line.
check: $(APP1_TARGETS)
	@for l in coalesced strided tiled; do \
	  for d in on off; do \
	    for s in always changed; do \
	      printf '%-10s dpx=%-3s store=%-7s ' "$$l" "$$d" "$$s"; \
	      ./$(APP1_TARGETS) --nodes $(CHECK_NODES) --trials 1 \
	        --warmup 0 --layout $$l --dpx $$d --store $$s \
	        | grep -v '^#' | tail -1; \
	    done; \
	  done; \
	done

# Runs the App1 sweep the paper reports, appending to one file per machine. It
# stops at the first failure rather than leaving a half-finished file that looks
# complete.
sweep: $(APP1_TARGETS)
	@if [ -n "$(GPU_MATCH)" ]; then \
	  name=$$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1); \
	  case "$$name" in \
	    *$(GPU_MATCH)*) echo "# node gpu: $$name" ;; \
	    *) echo "This node reports '$$name', not a $(GPU_MATCH). Refusing to"; \
	       echo "write $(SWEEP_CSV). Use the target for this machine, or pass"; \
	       echo "GPU_MATCH= to skip the check."; \
	       exit 1 ;; \
	  esac; \
	fi; \
	echo "# writing $(SWEEP_CSV)"; \
	for n in $(SWEEP_NODES); do \
	  for d in on off; do \
	    ./$(APP1_TARGETS) --nodes $$n --dpx $$d $(SWEEP_FLAGS) \
	      --trials $(SWEEP_TRIALS) --warmup $(SWEEP_WARMUP) \
	      --csv $(SWEEP_CSV) || exit 1; \
	  done; \
	done

# Machine shortcuts. Each fixes the architecture, the name tag, the output file
# and the GPU the sweep insists on, so nothing about one machine can reach the
# other. Adding a machine means one pair of lines here.
A100 = ARCHES=80 TAG=a100 GPU_MATCH=A100
H200 = ARCHES=90 TAG=h200 GPU_MATCH=H200

a100:       ; @$(MAKE) --no-print-directory $(A100) all
a100-app1:  ; @$(MAKE) --no-print-directory $(A100) app1
a100-app2:  ; @$(MAKE) --no-print-directory $(A100) app2
a100-check: ; @$(MAKE) --no-print-directory $(A100) check
a100-sweep: ; @$(MAKE) --no-print-directory $(A100) sweep
a100-clean: ; @$(MAKE) --no-print-directory $(A100) clean

h200:       ; @$(MAKE) --no-print-directory $(H200) all
h200-app1:  ; @$(MAKE) --no-print-directory $(H200) app1
h200-app2:  ; @$(MAKE) --no-print-directory $(H200) app2
h200-check: ; @$(MAKE) --no-print-directory $(H200) check
h200-sweep: ; @$(MAKE) --no-print-directory $(H200) sweep
h200-clean: ; @$(MAKE) --no-print-directory $(H200) clean

# Removes only this invocation's tag, so cleaning on one machine cannot delete
# the binaries another machine is measuring on a shared file system. Pass the
# same TAG you built with: make ARCHES=80 TAG=a100 clean.
clean:
	rm -f $(TARGETS)

# Removes builds for every tag, and the untagged binaries left by earlier
# versions of this file. Run it once before the first tagged build, and never
# while another machine is measuring, because it deletes that machine's
# binaries too.
clean-all:
	rm -f $(foreach t,$(BASE_TARGETS),$(t) $(t)-*)

help:
	@echo "Machine targets, which set everything for you:"
	@echo "  a100 h200              build both applications"
	@echo "  a100-app1 h200-app1    the network resilience system only"
	@echo "  a100-app2 h200-app2    the deep packet inspection system only"
	@echo "  a100-check h200-check  run the twelve routing variants"
	@echo "  a100-sweep h200-sweep  run the reported App1 sweep"
	@echo "  a100-clean h200-clean  remove that machine's binaries only"
	@echo ""
	@echo "General targets, which need ARCHES and TAG:"
	@echo "  all      build both applications (default)"
	@echo "  app1     build the network resilience system only"
	@echo "  app2     build the deep packet inspection system only"
	@echo "  check    build App1 and run its twelve kernel variants"
	@echo "  sweep    run the reported App1 sweep for this TAG"
	@echo "  clean    remove the binaries for this TAG only"
	@echo "  clean-all remove every tag (do not use while another machine runs)"
	@echo ""
	@echo "Variables and their current values:"
	@echo "  NVCC        = $(NVCC)"
	@echo "  ARCHES      = $(ARCHES)   (80 is the A100, 90 the H100 and H200)"
	@echo "  TAG         = $(TAG)"
	@echo "  NVCCFLAGS   = $(NVCCFLAGS)"
	@echo "  NVML_LIBS   = $(NVML_LIBS)"
	@echo "  CHECK_NODES = $(CHECK_NODES)"
	@echo "  SWEEP_NODES = $(SWEEP_NODES)"
	@echo "  SWEEP_CSV   = $(SWEEP_CSV)"
	@echo "  GPU_MATCH   = $(GPU_MATCH)"
	@echo ""
	@echo "Binaries this invocation would build:"
	@echo "  $(TARGETS)"
