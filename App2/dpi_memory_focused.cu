#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <nvml.h>         // NVML library header
#include <cuda.h>         // CUDA Driver API (though cuda_runtime.h is more common for kernels)
#include <cuda_runtime.h> // CUDA Runtime API

#include <stdbool.h> // For 'bool' type
#include <pthread.h> // For pthreads
#include <time.h>    // For timespec and clock_gettime

#ifndef HELPERS_H
#define HELPERS_H

#define PayloadSize 512
#define NumberOfSignatures 20000000
#define MaxSignatureLength 16
#define MatchingIndex 99501
#define CopyLen 14

#define midPoint 10000000

const int blockSize = 32; 
const int gridSize = (midPoint + blockSize - 1) / blockSize;


#define match 2
#define mismatch -1
#define indel -1

#define matchmismatch 1
#define indelmismatch -2

__constant__ char PktPayload_d_constant[PayloadSize];

#endif


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

// Function to analyze the power data file
void analyzePowerData(const char* filename, long long kernelStartTimeRelativeMs, long long kernelEndTimeRelativeMs, float kernelTimeMs) {
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

    // printf("\nAnalyzing power data between %lldms and %lldms (relative to polling start)...\n",
    //        kernelStartTimeRelativeMs, kernelEndTimeRelativeMs);

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
    // double kernelDurationSeconds = (double)(kernelEndTimeRelativeMs - kernelStartTimeRelativeMs) / 1000.0;
    float kernelDurationSeconds = kernelTimeMs / 1000.0; // Convert milliseconds to seconds
    double energyConsumedJoules = averagePower * kernelDurationSeconds;

    printf("Kernel Duration (measured by CUDA Event Library): %.6f ms\n", kernelTimeMs);
    printf("Number of power samples collected during kernel: %d\n", sampleCount);
    printf("Average Power during kernel: %.3lf Watts\n", averagePower);
    printf("Peak Power during kernel: %.3lf Watts\n", maxPower);
    printf("Minimum Power during kernel: %.3lf Watts\n", minPower);
    printf("Estimated Energy Consumed by kernel: %.3lf Joules\n", energyConsumedJoules);
}


#define CudaAllocErrorCheck \
    if (err != cudaSuccess) { \
        fprintf(stderr, "cudaMalloc failed: %s\\n", cudaGetErrorString(err)); \
        return -1; \
    }

char* generate_string(int str_len){

    char* str = (char*)malloc(str_len + 1);
    if (!str) return NULL;

    for(int i = 0; i< str_len;i++){
        char rand_char = rand()%26 + 97;
        str[i] = rand_char;
    }
    str[str_len] = '\0';

    return str;
}

void generate_signatures(FILE* f){

   for(int i = 0; i< NumberOfSignatures; i++){
    char* sig = generate_string(MaxSignatureLength);
    fprintf(f,"%s\n",sig);
    free(sig);
   } 

}

#define thread_id (blockDim.x*blockDim.y)*threadIdx.z+(threadIdx.y*blockDim.x)+(threadIdx.x)

#define block_id (gridDim.x*gridDim.y)*blockIdx.z+(blockIdx.y*gridDim.x)+(blockIdx.x)

#define block_size blockDim.x *blockDim.y*blockDim.z

__global__ void 
construct_trace_matrix_gpu_low(uint32_t *traceMatrix, const char* signatures,int32_t *report){

    // __shared__ int midPoint;
    // midPoint = NumberOfSignatures/2;

    // __shared__ int64_t row_size;
    int64_t row_size = (MaxSignatureLength+1)*midPoint;


    int32_t global_thread_id = block_id * block_size + thread_id;
    
    if(global_thread_id > midPoint){
        return;
    }

    int32_t signatures_offset = (global_thread_id) * (MaxSignatureLength + 1)-1;

    int32_t signatures_offset_1 = (global_thread_id + midPoint) * (MaxSignatureLength + 1)-1;

    char thread_signature[MaxSignatureLength+1];
    char thread_signature_1[MaxSignatureLength+1];
    
    
    for(int j =1; j< MaxSignatureLength+1;j++){
        thread_signature[j] = signatures[signatures_offset + j];
        thread_signature_1[j] = signatures[signatures_offset_1 + j];  
        
    }
    
    uint32_t row1[MaxSignatureLength+1];
    uint32_t row2[MaxSignatureLength+1];


    for(int j =0; j< MaxSignatureLength+1;j++){
        row1[j] = 0;
        row2[j] = 0;
    }

    uint32_t north, north_west, west,temp_hi, temp_low;


    bool even = 0;

    uint32_t current_value;

    for(int i = 2; i < PayloadSize+1; i ++){

        uint32_t iter_index = global_thread_id + row_size;
        
        even = !even;

        for(int j = 1; j < MaxSignatureLength+1; j ++){

            iter_index += midPoint;

            if(even){
                north_west = row1[j-1];
                north = row1[j];
                west = row2[j-1];
            }
            else {
                north_west = row2[j-1];
                north = row2[j];
                west = row1[j-1];
            }
            
            current_value = __vimax3_s16x2_relu(north,north_west,west);

            temp_hi = current_value &(0x0000FFFF);
            temp_low = current_value>>16;

            bool a =  PktPayload_d_constant[i-1] == thread_signature[j];
            bool b =  PktPayload_d_constant[i-1] == thread_signature_1[j];

            current_value = (temp_hi + (matchmismatch) * (a) + (indelmismatch)* (!a)*(temp_hi>1) ) 
            | ((temp_low + (matchmismatch)  * (b)+ (indelmismatch)* (!b)*(temp_low>1))<<16);


            if(even){
                row2[j] = current_value;
            }
            else {
                row1[j] = current_value;
            }

            if (temp_hi>0.8*MaxSignatureLength){
                report[0] = temp_hi;
                report[1] = global_thread_id;
                return;
            }
            else if (temp_low >0.8*MaxSignatureLength){
                report[0] = temp_low;
                report[1] = global_thread_id + midPoint;
                return;
            }
        }
    }
}

int main(int argc, char* argv[]) {
    srand(time(NULL));

    // Configuration for power monitoring
    const unsigned int targetDeviceIdx = 0; // Monitor the first GPU
    const char* powerLogFilename = "DPI_Power_Log.csv";
    // Poll as fast as possible:
    // 1ms is a common choice; 0ms makes the thread spin and yield, consuming more CPU
    // but ensures minimum delay between NVML calls (won't get new data faster than hardware allows).
    const long long pollRateMs = 1;


    char* gpu_type = (char*)malloc(10 * sizeof(char));
    if (argc > 1) {
        gpu_type = argv[1];
    }

    int gpu_id = 0;
    cudaSetDevice(gpu_id);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, gpu_id);
    printf("Using GPU %d: %s\n", gpu_id, prop.name);
    printf("Max global memory: %zu bytes\n", prop.totalGlobalMem);

    FILE *f;

    f = fopen("signatures.txt","w");
    if(f){
        generate_signatures(f);
        fclose(f);
    }

    char *PktPayload = generate_string(PayloadSize);

    size_t totalBytes = NumberOfSignatures * (MaxSignatureLength+1) * sizeof(char);
    char* Signatures = (char*)malloc(totalBytes);

    if (!Signatures) {
        fprintf(stderr, "Error: failed to allocate memory\n");
        exit(1);
    }

    f = fopen("signatures.txt", "r");
    if (!f) {
        perror("Failed to open signatures.txt");
        exit(1);
    }

    char buffer[MaxSignatureLength+2];
    int index = 0;

    while (fgets(buffer, sizeof(buffer), f) && index < NumberOfSignatures) {
        buffer[strcspn(buffer, "\n")] = '\0';
        strcpy(Signatures + index * (MaxSignatureLength+1), buffer);
        index++;
    }

    strncpy(PktPayload, Signatures+MatchingIndex*(MaxSignatureLength+1),CopyLen);
    fclose(f);

    char *PktPayload_d;
    char *Signatures_d;

    size_t traceMatrixSize = sizeof(uint32_t)*(MaxSignatureLength+1)*((NumberOfSignatures+1)/2)*3* sizeof(char);

    uint32_t *traceMatrix = (uint32_t*)malloc(traceMatrixSize);
    if (traceMatrix) memset(traceMatrix, 0, traceMatrixSize);
    uint32_t *traceMatrix_d;

    int32_t *report = (int32_t*)malloc(2*sizeof(int32_t));
    report[0] = 0;
    report[1] = 0;
    int32_t *report_d;

    cudaError_t err;

    err = cudaMalloc((void**)&PktPayload_d, sizeof(char) * PayloadSize);
    CudaAllocErrorCheck;

    err = cudaMalloc((void**)&Signatures_d, sizeof(char) * totalBytes);
    CudaAllocErrorCheck;

    err = cudaMalloc((void**)&report_d, sizeof(int32_t) * 2);
    CudaAllocErrorCheck;


    cudaMemcpy(PktPayload_d, PktPayload, sizeof(char) * PayloadSize, cudaMemcpyHostToDevice);
    cudaMemcpy(report_d, report, 2*sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(PktPayload_d_constant, PktPayload, PayloadSize);

    cudaEvent_t start_event, stop_event;
    float kernel_time_ms = 0.0f;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);

    u_int32_t* NumberOfSignatures_h = (u_int32_t*)malloc(sizeof(u_int32_t));
    *NumberOfSignatures_h = NumberOfSignatures;
    u_int32_t* NumberOfSignatures_d;
    cudaMalloc((void**)&NumberOfSignatures_d, sizeof(u_int32_t));

    u_int32_t* RowSize_h = (u_int32_t*)malloc(sizeof(u_int32_t));
    u_int32_t* RowSize_d;
    cudaMalloc((void**)&RowSize_d, sizeof(u_int32_t));

    size_t free_mem, total_mem;
    err = cudaMemGetInfo(&free_mem, &total_mem);

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

    struct timespec ts_kernel_launch;
    clock_gettime(CLOCK_MONOTONIC, &ts_kernel_launch);
    // Calculate kernel start time relative to the overall measurement start
    long long kernelStartTimeRelativeMs = (ts_kernel_launch.tv_sec * 1000 + ts_kernel_launch.tv_nsec / 1000000) - overallStartTimeMs;

    if(free_mem > sizeof(int16_t) * totalBytes* 3) {

        printf("Tracing matrix size is %lld MB\n",traceMatrixSize/1000000);

        printf("Cuda memory available: %lld bytes\n", free_mem);

        cudaMemcpy(Signatures_d, Signatures, sizeof(char) * totalBytes, cudaMemcpyHostToDevice);
        err = cudaMalloc((void**)&traceMatrix_d, traceMatrixSize);
        CudaAllocErrorCheck;
        cudaMemcpy(traceMatrix_d, traceMatrix, traceMatrixSize, cudaMemcpyHostToDevice);
        cudaMemcpy(NumberOfSignatures_d, NumberOfSignatures_h, sizeof(u_int32_t), cudaMemcpyHostToDevice);

        cudaEventRecord(start_event, 0);

        construct_trace_matrix_gpu_low<<<dim3(gridSize +1,1,1),dim3(blockSize,1,1)>>>(traceMatrix_d, Signatures_d,report_d);

    } 

    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&kernel_time_ms, start_event, stop_event);

    cudaMemcpy(report, report_d, 2*sizeof(u_int32_t), cudaMemcpyDeviceToHost);

    printf("time taken by the GPU kernel is %.6f s\n", kernel_time_ms/1000);

    if(report[0] > 0){
        printf("Signature matching occured on signature %u with %u matching score\n",report[1],report[0]);
    }
    else{
        printf("No matching has been detecting\n");
    }

    struct timespec ts_kernel_complete;
    clock_gettime(CLOCK_MONOTONIC, &ts_kernel_complete);
    // Calculate kernel end time relative to the overall measurement start
    long long kernelEndTimeRelativeMs = (ts_kernel_complete.tv_sec * 1000 + ts_kernel_complete.tv_nsec / 1000000) - overallStartTimeMs;
    // --- END KERNEL EXECUTION WINDOW ---

    // Give the polling thread a moment to capture final data points after kernel completion
    nanosleep(&initial_delay, NULL);

    // Stop NVML power polling
    nvmlAPIEnd();
    // --- END POWER MEASUREMENT ---


    // Analyze the collected power data
    analyzePowerData(powerLogFilename, kernelStartTimeRelativeMs, kernelEndTimeRelativeMs, kernel_time_ms);

    free(PktPayload);
    free(Signatures);

    cudaFree(PktPayload_d);
    cudaFree(Signatures_d);
    cudaFree(traceMatrix_d);

    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);

    return 0;
}