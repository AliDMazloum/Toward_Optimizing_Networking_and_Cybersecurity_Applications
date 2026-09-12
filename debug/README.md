# debug

Scratch tooling for checking the measurement path, shared here because it has
to run on the machines that own the hardware and they all reach this
repository. Nothing in this folder is part of either application, nothing in
the applications depends on it, and the folder is expected to be deleted once
the questions that prompted it are settled.

## validate_gpu_power.sh

Checks what the NVML power path in these programs actually measures. The energy
they report comes from a sampling thread calling `nvmlDeviceGetPowerUsage`
while the kernel runs, and three separate faults make that number wrong while
looking the same in the output: the sampler reading a card that is not running
the kernel, because CUDA and NVML order devices differently on a multi-GPU
machine; a driver value refreshed far more slowly than the sampler polls; or a
kernel that genuinely draws little because it leaves the device far below its
power limit. The script runs one job, watches it from outside with
`nvidia-smi`, and compares that against what the program reports for the same
window, which tells the three apart.

Run it from the root of the clone, on a machine that has the GPU:

    make a100-app1
    ./debug/validate_gpu_power.sh App1/floyd_warshall_routing-a100 24000 0

The three arguments are the binary, the problem size and the device index; they
default to the A100 binary, 24000 nodes and device 0. Pass the same binary,
size and device index as the run being checked. Its closing section states what
each outcome means. It writes only into a temporary directory of its own and
changes nothing on the machine.

## run_energy_sweeps.sh

Hands a machine the whole energy job in one command: build, characterise the
power instrument with the script above, then run the App1 and App2 energy
sweeps, logging each stage.

    ./debug/run_energy_sweeps.sh a100
    ./debug/run_energy_sweeps.sh h200
    ./debug/run_energy_sweeps.sh h200 --check-only

The characterisation is not a formality, because two of its outputs decide how
the energy may afterwards be reported: the idle draw of the card the kernel
actually uses, which is the floor every short run reports and the baseline that
load draw has to be read against, and how often the driver refreshes its power
reading, which sets the shortest run whose energy means anything. A run that
finishes inside one refresh reports a value the driver computed before the run
began. Both differ by machine and driver, so they are measured rather than
assumed, and the refresh interval has to be carried back into the analysis on
the workstation before any of the numbers are quoted.

Sweeps append, and the analysis keeps the last five trials of each
configuration, so re-running after an interruption neither duplicates work nor
repeats a configuration that already finished.

## check_csv_guard.sh

Exercises the header guard in both programs. A results csv is appended to
across many runs, and a binary built from a different commit writes a different
set of columns; appending under a header that describes the older set leaves
every reader one field out of step, and nothing reports it, because a csv
carries no statement of how many columns a row should have. The file still
parses and the numbers land under the wrong names. Both programs now compare
the header they would write against the one already in the file and stop before
the run rather than after the work is done.

That guard is cheap to get wrong in ways a build does not catch, so it is
tested rather than assumed: a constant matching nothing would refuse every
append from now on, and one matching too loosely would let the original problem
back in.

    ./debug/check_csv_guard.sh App1/floyd_warshall_routing-a100 --cpu --nodes 200
    ./debug/check_csv_guard.sh App2/smith_waterman_dpi-a100 --cpu --signatures 2000

Build with whichever invocation that machine uses (`make a100-app1`,
`make h200-app2`, or a `TAG=` of its own) and give the script the binary it
produced. The cases run on the CPU path, so any build works and no GPU is
needed. Give it the smallest configuration the program accepts, because the
point is the csv rather than the measurement. It checks three cases: a file that does
not exist is created with its header, a file carrying the right header is
appended to, and a file whose header is one column short is refused with the
file left byte for byte unchanged. It works inside a temporary directory and
touches nothing else.

## run_scalefree_app1.sh

Sweeps App1 over the generated Barabasi-Albert graph on one machine,
unattended, in one command.

    ./debug/run_scalefree_app1.sh h200
    ./debug/run_scalefree_app1.sh a100
    ./debug/run_scalefree_app1.sh epyc

Each stage is independent and a failure in one does not stop the rest, because
this is meant to be started and left. The summary at the end says which stages
produced a file and how many rows, so a partial session is obvious rather than
silent, and a stage that stopped leaves the rows it had already written, which
are sound. Re-running appends rather than repeating what finished.

Verification is switched off on the CPU stages, deliberately: on that path the
reference and the measurement are the same host triple loop on the same input,
so the check would compare it against itself. The GPU stages keep it on, and on
a generated graph, which has no closed-form distance oracle, that check runs the
host loop once per invocation. That is where the correctness evidence for this
application comes from, and it costs about 100 s per invocation at 24,000 nodes.

## run_overhead_control.sh

Measures what the current App2 program costs against the original one, at the
single configuration where the two can be compared. The original's problem size
is fixed at compile time; the current program takes its size at run time, so the
script reads the original's constants out of its source and runs both arms at
those values. Setting the size in two places would let them drift apart, and a
control that compares one size against another reports a ratio that is mostly
the size difference.

    ./debug/run_overhead_control.sh a100
    ./debug/run_overhead_control.sh h200

Since the current program moved the character score inside the maximum, onto
the diagonal only, the two arms compute different recurrences, so this control
now measures the cost of the program's structure and its extra arithmetic
together and no longer isolates the first.

This matters because every ratio taken between two cards assumes the same
program was measured on both. If the current program costs a few percent on one
card and a great deal on the other, that difference lands in the ratio and reads
as a property of the hardware.

The comparison needs care, because several effects can masquerade as each other.
The original has no trial loop and its warm-up is commented out, so it measures
a cold launch, and comparing that against a warm mean would charge the clock
state to the program. It also launches 64 threads per block where the sweep uses
32, so a block-size difference would be charged to the program too. And the
original's problem size is a compile-time constant, so its compiler knows the
payload loop's trip count and the detection threshold, while the current
program takes both at run time. Five arms separate those: the original cold, the
current cold at the original's block size, the current cold at the swept block
size, the current under the swept protocol, and the current built with the
`PIN_` macros that put the run-time parameters back into constants. The first
ratio is the one the comparison rests on; the others say what the remaining
difference is made of.

The script also prints what ptxas reports for both sources, registers per thread
and any spill to local memory, because a kernel that spills is slower for a
reason no timing can show.

The original writes a signatures file of about 160 MB into its working directory
on every run, so each arm runs in a scratch directory that is deleted at the
end. Nothing is appended to any csv.

## run_detection_quality.sh

Measures how often the detector fires on a payload carrying a planted signature
and how often it fires on one that does not, as a curve over the threshold and
over the alphabet the synthetic data is drawn from.

    ./debug/run_detection_quality.sh h200
    SEEDS=20 ./debug/run_detection_quality.sh h200      a short rehearsal first

The alphabet is swept rather than fixed because two unrelated strings agree at
one position with probability one over the alphabet size, so a false positive
rate over 26 lowercase letters is roughly ten times the rate over the whole byte
range. Quoting one alphabet would state a result about the generator rather than
about the detector; quoting three states what a change of traffic would do, which
is the part that carries over to traffic nobody generated.

One run contributes one payload and one bit, so every rate is a count over seeds
and its resolution is one over the seed count. The alpha loop cannot be
collapsed: the scores do not depend on the threshold but the decision does, and
the program reports the first crossing rather than the largest score, so a
crossing at a low threshold says nothing about a high one.

Every run keeps `--verify report` on, so if the device and the host reference
ever disagreed about a score the mismatch column would say so in the data rather
than the rates quietly being wrong.

It refuses to overwrite its output and appends to nothing. Unlike the sweeps it
does not refuse a busy card, deliberately: it measures which payloads are
detected rather than how long anything takes, and a detection is the same
detection on a shared card. `SWEEP_GPU` still pins it out of the way. It prints
the confusion matrix at the end, and a closing section saying how to read it,
including why the true positive rate in regex mode is 1 at every threshold by
construction.

## check_regex_exit.sh

Decides why one sweep came out slower than an earlier sweep of the same
configuration: because of how the early-exit predicate is compiled, or because
something else was using the card at the time.

    ./debug/check_regex_exit.sh a100
    ./debug/check_regex_exit.sh h200

The two explanations are indistinguishable in a results file, which records the
configuration and nothing about what else the machine was doing, so the
slowdown has to be reproduced with the two causes separated. One arm passes the
predicate as a kernel argument and the other builds it into a constant with
`-DPIN_EXIT_FIRST=1`; one source builds both, so nothing is checked out.

The arms are interleaved rather than run one after the other, because a
neighbouring job that comes and goes would otherwise land on whichever arm
happened to be running and would read exactly like a property of the code. What
else holds memory on the card is recorded before and after as well, since a
neighbour present for the whole run is invisible to the interleaving.

It also prints the register count of the regex kernel each arm launches, which
bounds what the code can be responsible for whatever the timings do. Equal
counts settle the question outright, because identical machine code cannot run
at two speeds for a reason inside the code, and counts a few apart with neither
arm spilling cannot produce a large factor either.

It refuses to start on a card another job is computing on, because a timing
measurement taken beside somebody else's job reports the sharing rather than
the program, and it says so instead of measuring through. On a shared node,
name a free card with `SWEEP_GPU`, the same variable the sweeps take:

    nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv
    SWEEP_GPU=3 ./debug/check_regex_exit.sh a100

The card is pinned by its UUID rather than by an index, because CUDA and NVML
number the devices differently and an index that names the free card to
nvidia-smi can name a busy one to the program. Both applications bind their
power sampler by PCI bus id taken from the CUDA device, so the energy path
follows the same pinning. With `SWEEP_GPU` unset the program picks its own
device and no index can be checked, so the whole node has to be free.

`ALLOW_BUSY=1` overrides the refusal, which is worth doing only to demonstrate
that a busy card is the cause of something. A job that arrives after the start
is caught between arms and reported as a warning.

The closing section states what each outcome means. It writes no csv and
removes its two binaries at the end.
