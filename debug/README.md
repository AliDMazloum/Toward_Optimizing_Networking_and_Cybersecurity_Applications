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
