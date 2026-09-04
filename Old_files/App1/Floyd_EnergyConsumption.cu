#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <stdbool.h> // For 'bool' type
#include <pthread.h> // For pthreads
#include <time.h>    // For timespec and clock_gettime

#include <nvml.h>         // NVML library header
#include <cuda.h>         // CUDA Driver API (though cuda_runtime.h is more common for kernels)
#include <cuda_runtime.h> // CUDA Runtime API

// --- Global Constants and Variables ---
#define Ver 24000
const int V = Ver;
const int INF = 99999;

// CUDA grid and block dimensions
int blockSize = 256;
int gridSize = 4096; // This will be the number of blocks in X dimension

// NVML Power Monitoring Global Variables
volatile bool pollThreadStatus = false; // Use volatile for thread flag
nvmlDevice_t nvmlDeviceID;
pthread_t powerPollThread;
FILE *powerLogFile = NULL; // File pointer for power data
nvmlReturn_t nvmlResult;   // Global NVML return variable

// Structure to pass arguments to the polling thread
typedef struct {
    long long pollIntervalMs;
} PowerPollingArgs;

// --- CUDA Error Checking Macro ---
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// --- NVML Power Monitoring Functions ---

/*
Error checking function for NVML API calls.
Prints an error message and exits on failure.
*/
void checkNVMLError(nvmlReturn_t resultToCheck, const char* functionName)
{
    if (NVML_SUCCESS != resultToCheck) {
        fprintf(stderr, "NVML Error in %s: %s\n", functionName, nvmlErrorString(resultToCheck));
        // Clean up NVML and file before exiting
        if (powerLogFile != NULL) {
            fclose(powerLogFile);
            powerLogFile = NULL;
        }
        nvmlShutdown();
        exit(EXIT_FAILURE);
    }
}

/*
Polling function for the separate thread.
Continuously queries GPU power usage and logs it with a timestamp.
*/
void *powerPollingFunc(void *ptr)
{
    unsigned int powerLevel = 0; // in milliWatts
    nvmlEnableState_t pmmode;
    PowerPollingArgs *args = (PowerPollingArgs*)ptr;
    long long pollIntervalMs = args->pollIntervalMs;

    // Use clock_gettime for high-resolution, monotonic timestamps
    struct timespec ts_start;
    clock_gettime(CLOCK_MONOTONIC, &ts_start);
    long long startTimeMs = ts_start.tv_sec * 1000 + ts_start.tv_nsec / 1000000;

    fprintf(powerLogFile, "Timestamp_ms,Power_Watts\n"); // CSV Header

    while (pollThreadStatus)
    {
        // Disable thread cancellation during critical operations (like file I/O)
        pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, NULL);

        // Get the power management mode of the GPU.
        nvmlResult = nvmlDeviceGetPowerManagementMode(nvmlDeviceID, &pmmode);
        // Error handling for polling loop: print error but try to continue
        if (NVML_SUCCESS != nvmlResult) {
            fprintf(stderr, "NVML Warning in powerPollingFunc (nvmlDeviceGetPowerManagementMode): %s\n", nvmlErrorString(nvmlResult));
        } else {
            // Get the power usage in milliWatts.
            nvmlResult = nvmlDeviceGetPowerUsage(nvmlDeviceID, &powerLevel);
            if (NVML_SUCCESS != nvmlResult) {
                 fprintf(stderr, "NVML Warning in powerPollingFunc (nvmlDeviceGetPowerUsage): %s\n", nvmlErrorString(nvmlResult));
            }
        } 

        struct timespec ts_current;
        clock_gettime(CLOCK_MONOTONIC, &ts_current);
        long long currentTimeMs = ts_current.tv_sec * 1000 + ts_current.tv_nsec / 1000000;
        long long elapsedMs = currentTimeMs - startTimeMs; // Time relative to polling start

        // The output file stores power in Watts.
        fprintf(powerLogFile, "%lld,%.3lf\n", elapsedMs, (double)powerLevel / 1000.0);
        fflush(powerLogFile); // Ensure data is written to disk promptly

        // Re-enable thread cancellation
        pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, NULL);

        // Introduce a delay for polling. nanosleep is more precise than sleep().
        if (pollIntervalMs > 0) {
            struct timespec sleep_ts;
            sleep_ts.tv_sec = pollIntervalMs / 1000;
            sleep_ts.tv_nsec = (pollIntervalMs % 1000) * 1000000;
            nanosleep(&sleep_ts, NULL);
        } else {
            // If pollIntervalMs is 0, spin as fast as possible (can consume more CPU)
            // yield to other threads if possible
            sched_yield();
            // pthread_yield();
        }
    }

    pthread_exit(NULL); // Thread exits gracefully
}

/*
Initializes NVML, selects the target GPU, opens the log file,
and spawns the power polling thread.
*/
void nvmlAPIRun(unsigned int targetDeviceIndex, long long pollIntervalMs, const char* filename)
{
    unsigned int deviceCount = 0;
    char deviceNameStr[NVML_DEVICE_NAME_BUFFER_SIZE]; // Use NVML defined size

    // Initialize nvml.
    nvmlResult = nvmlInit();
    checkNVMLError(nvmlResult, "nvmlInit");

    // Count the number of GPUs available.
    nvmlResult = nvmlDeviceGetCount(&deviceCount);
    checkNVMLError(nvmlResult, "nvmlDeviceGetCount");

    if (targetDeviceIndex >= deviceCount) {
        fprintf(stderr, "Error: Device index %u is out of range (only %u devices found).\n", targetDeviceIndex, deviceCount);
        checkNVMLError(NVML_ERROR_INVALID_ARGUMENT, "Invalid Device Index"); // This will shut down NVML and exit
    }

    // Get the device ID for the target device.
    nvmlResult = nvmlDeviceGetHandleByIndex(targetDeviceIndex, &nvmlDeviceID);
    checkNVMLError(nvmlResult, "nvmlDeviceGetHandleByIndex");

    // Get the name of the device.
    nvmlResult = nvmlDeviceGetName(nvmlDeviceID, deviceNameStr, sizeof(deviceNameStr));
    checkNVMLError(nvmlResult, "nvmlDeviceGetName");
    printf("Monitoring device: %s (Index: %u)\n", deviceNameStr, targetDeviceIndex);

    // Open the power log file for writing (truncates if exists)
    powerLogFile = fopen(filename, "w");
    if (powerLogFile == NULL) {
        perror("Failed to open power log file");
        checkNVMLError(NVML_ERROR_UNKNOWN, "fopen"); // This will shut down NVML and exit
    }

    pollThreadStatus = true; // Signal the polling thread to start its loop

    // Create a static struct to hold args for the thread, so its address is stable
    static PowerPollingArgs threadArgs;
    threadArgs.pollIntervalMs = pollIntervalMs;

    int iret = pthread_create(&powerPollThread, NULL, powerPollingFunc, (void*)&threadArgs);
    if (iret)
    {
        fprintf(stderr,"Error - pthread_create() return code: %d\n", iret);
        checkNVMLError(NVML_ERROR_UNKNOWN, "pthread_create"); // This will shut down NVML and exit
    }
    printf("Power polling thread started.\n");
}

/*
Signals the polling thread to stop and waits for its termination.
Closes the log file and shuts down NVML.
*/
void nvmlAPIEnd()
{
    if (!pollThreadStatus) {
        printf("Power polling is not active or already stopped.\n");
        return;
    }

    pollThreadStatus = false; // Signal the polling thread to exit its loop
    pthread_join(powerPollThread, NULL); // Wait for the polling thread to finish

    printf("Power polling thread stopped.\n");

    if (powerLogFile != NULL) {
        fclose(powerLogFile); // Close the file
        powerLogFile = NULL;
    }

    nvmlResult = nvmlShutdown();
    checkNVMLError(nvmlResult, "nvmlShutdown");
}

// --- CUDA Kernel and Helper Functions ---

// Kernel function for inner two loop of Floyd Warshall Algorithm
__global__
void FloydWarshall_gpu(const int V_param, const int k, int *dis)
{
    // int t = (blockDim.x * blockDim.y) * threadIdx.z + (threadIdx.y * blockDim.x) + (threadIdx.x);
    // int b = (gridDim.x * gridDim.y) * blockIdx.z + (blockIdx.y * gridDim.x) + (blockIdx.x);
    // int T = blockDim.x * blockDim.y * blockDim.z; // Total threads per block
    // int B = gridDim.x * gridDim.y * gridDim.z;   // Total blocks in grid

    // // int tm;

    // for (int i = b; i < V_param; i += B)
    // {
    //     for (int j = t; j < V_param; j += T)
    //     {
    //         // tm = dis[i * V_param + k] + dis[k * V_param + j];
    //         dis[i * V_param + j] = __viaddmin_s32(dis[i * V_param + k], dis[k * V_param + j], dis[i * V_param + j]);
    //         // dis[i * V_param + j] = tm * (tm < dis[i * V_param + j]) + dis[i * V_param + j] * (tm >= dis[i * V_param + j]);
    //     }
    // }

    int thread_id = (blockDim.x*blockDim.y)*threadIdx.z+(threadIdx.y*blockDim.x)+(threadIdx.x);
    //BLock ID
    int block_id =(gridDim.x*gridDim.y)*blockIdx.z+(blockIdx.y*gridDim.x)+(blockIdx.x);
    //Block size
    int block_size = blockDim.x *blockDim.y*blockDim.z;
    //grid size

    int global_thread_id = block_id * block_size + thread_id;
	int total_number_of_threads = gridDim.x*gridDim.y*gridDim.z*block_size;

	for(int64_t i = global_thread_id; i < V*V; i +=total_number_of_threads){
		int row = i%V;
		int coloumn = i/V;
        dis[row * V + coloumn] = __viaddmin_s32(dis[row*V+k], dis[k*V+coloumn], dis[row * V + coloumn]);
		// int row = i%V;
		// int coloumn = i/V;
        // dis[i] = __viaddmin_s32(dis[row*V+k], dis[k*V+coloumn], dis[i]);
    }

    // for (int i=blockIdx.x;i<V;i+=gridDim.x)
    // {
	// 	int temp = dis[i*V+k];
    //     bool val;
    //     int result;
	// 	  for(int j=threadIdx.x;j<V;j+=blockDim.x)
	// 	  {
    //             result = __vibmin_s32(temp + dis[k*V+j], dis[i*V+j],&val);
    //             if(val){
    //                 dis[i*V+j] = result;
    //             }

    //             // if(temp + dis[k*V +j] < dis[i*V+j] ){
    //             //     dis[i*V+j] = temp + dis[k*V +j];
    //             // }
    //         // }
	// 	  }
	// }
}

// Function to run your CUDA kernel and ensure completion
void runFloydWarshallKernel(int V_param, int *dis_d, dim3 grid_dim, dim3 block_dim) {
    for (int k = 0; k < V_param; k++)
    {
        FloydWarshall_gpu<<<grid_dim, block_dim>>>(V_param, k, dis_d);
        // CRUCIAL: Synchronize after EACH kernel launch to ensure it completes before next iteration.
        // This is important for correctness of Floyd-Warshall's iterative nature.
        // For power measurement, it also ensures the GPU is active during the loop.
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

// Function to analyze the power data file
void analyzePowerData(const char* filename, long long kernelStartTimeRelativeMs, long long kernelEndTimeRelativeMs) {
    FILE *fp = fopen(filename, "r");
    if (fp == NULL) {
        perror("Error: Could not open power data file for analysis");
        return;
    }

    char line[256];
    // Read and discard header line
    if (fgets(line, sizeof(line), fp) == NULL) {
        fprintf(stderr, "Error: Could not read header from power data file.\n");
        fclose(fp);
        return;
    }

    double totalPower = 0.0;
    int sampleCount = 0;
    double maxPower = 0.0;
    double minPower = 1000000.0; // Initialize with a very high value

    long long timestamp; // Timestamp from the file (relative to polling start)
    double power;
    char comma;

    printf("\nAnalyzing power data between %lldms and %lldms (relative to polling start)...\n",
           kernelStartTimeRelativeMs, kernelEndTimeRelativeMs);

    while (fscanf(fp, "%lld%c%lf\n", &timestamp, &comma, &power) == 3) {
        // Only consider samples within the kernel's execution window
        if (timestamp >= kernelStartTimeRelativeMs && timestamp <= kernelEndTimeRelativeMs) {
            totalPower += power;
            sampleCount++;
            if (power > maxPower) maxPower = power;
            if (power < minPower) minPower = power;
        }
    }

    fclose(fp);

    if (sampleCount == 0) {
        printf("No power samples found within the kernel execution window.\n");
        return;
    }

    double averagePower = totalPower / sampleCount;
    // Calculate kernel duration in seconds for energy calculation
    double kernelDurationSeconds = (double)(kernelEndTimeRelativeMs - kernelStartTimeRelativeMs) / 1000.0;
    double energyConsumedJoules = averagePower * kernelDurationSeconds;

    printf("Kernel Duration (measured by CPU clock): %lld ms\n", kernelEndTimeRelativeMs - kernelStartTimeRelativeMs);
    printf("Number of power samples collected during kernel: %d\n", sampleCount);
    printf("Average Power during kernel: %.3lf Watts\n", averagePower);
    printf("Peak Power during kernel: %.3lf Watts\n", maxPower);
    printf("Minimum Power during kernel: %.3lf Watts\n", minPower);
    printf("Estimated Energy Consumed by kernel: %.3lf Joules\n", energyConsumedJoules);
}

// --- Main Function ---
int main(void)
{

    // Configuration for power monitoring
    const unsigned int targetDeviceIdx = 0; // Monitor the first GPU
    const char* powerLogFilename = "FloydWarshall_Power_Log.csv";
    // Poll as fast as possible:
    // 1ms is a common choice; 0ms makes the thread spin and yield, consuming more CPU
    // but ensures minimum delay between NVML calls (won't get new data faster than hardware allows).
    const long long pollRateMs = 1;

    int *dis = (int*)malloc(V * V * sizeof(int));
    int *dis_d;

    CUDA_CHECK(cudaMalloc((void**)&dis_d, V * V * sizeof(int)));

    // Initialize dis array on the host
    for (int i = 0; i < V; i++)
    {
        for(int j = 0; j < V; j++)
        {
            if(j == i + 1) dis[i * V + j] = 1;
            else if (i != j) dis[i * V + j] = INF;
            else dis[i * V + j] = 0;
        }
    }

    // --- START POWER MEASUREMENT ---
    // Get the global start time for relative timestamps
    struct timespec ts_overall_start;
    clock_gettime(CLOCK_MONOTONIC, &ts_overall_start);
    long long overallStartTimeMs = ts_overall_start.tv_sec * 1000 + ts_overall_start.tv_nsec / 1000000;

    // Start NVML power polling thread
    nvmlAPIRun(targetDeviceIdx, pollRateMs, powerLogFilename);

    // Give the polling thread a moment to initialize and write its first timestamp.
    // This ensures that the kernel's start time can find corresponding entries in the log.
    struct timespec initial_delay = { .tv_sec = 0, .tv_nsec = 100 * 1000000 }; // 100 milliseconds
    nanosleep(&initial_delay, NULL);


    // --- KERNEL EXECUTION WINDOW ---
    printf("\nLaunching Floyd-Warshall kernel on GPU (V=%d)...\n", V);

    struct timespec ts_kernel_launch;
    clock_gettime(CLOCK_MONOTONIC, &ts_kernel_launch);
    // Calculate kernel start time relative to the overall measurement start
    long long kernelStartTimeRelativeMs = (ts_kernel_launch.tv_sec * 1000 + ts_kernel_launch.tv_nsec / 1000000) - overallStartTimeMs;

    // Copy initial data to device
    CUDA_CHECK(cudaMemcpy(dis_d, dis, V * V * sizeof(int), cudaMemcpyHostToDevice));

    // Define CUDA grid and block dimensions
    dim3 grid_dim(gridSize, 1, 1);
    dim3 block_dim(blockSize, 1, 1);

    // Run the main Floyd-Warshall kernel loop
    runFloydWarshallKernel(V, dis_d, grid_dim, block_dim);

    struct timespec ts_kernel_complete;
    clock_gettime(CLOCK_MONOTONIC, &ts_kernel_complete);
    // Calculate kernel end time relative to the overall measurement start
    long long kernelEndTimeRelativeMs = (ts_kernel_complete.tv_sec * 1000 + ts_kernel_complete.tv_nsec / 1000000) - overallStartTimeMs;

    printf("Floyd-Warshall kernel completed.\n");
    printf("GPU Kernel execution time: %.6f seconds\n",
           (double)(kernelEndTimeRelativeMs - kernelStartTimeRelativeMs) / 1000.0);
    // --- END KERNEL EXECUTION WINDOW ---

    // Give the polling thread a moment to capture final data points after kernel completion
    nanosleep(&initial_delay, NULL);

    // Stop NVML power polling
    nvmlAPIEnd();
    // --- END POWER MEASUREMENT ---


    // Copy results back from device to host
    CUDA_CHECK(cudaMemcpy(dis, dis_d, V * V * sizeof(int), cudaMemcpyDeviceToHost));

    // Verification (your original assert loop)
    printf("\nVerifying results...\n");
    for (int i = 0; i < V; i++)
    {
        for(int j = 0; j < V; j++)
        {
            if(j >= i)
            {
                assert(dis[i * V + j] == j - i);
            }
            else assert(dis[i * V + j] == INF);
        }
    }
    printf("Verification successful.\n");

    // Analyze the collected power data
    analyzePowerData(powerLogFilename, kernelStartTimeRelativeMs, kernelEndTimeRelativeMs);

    // Free memory
    free(dis);      // Free host memory
    CUDA_CHECK(cudaFree(dis_d)); // Free device memory

    return 0;
}