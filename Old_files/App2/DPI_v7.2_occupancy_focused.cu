#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

#ifndef HELPERS_H
#define HELPERS_H

#define PayloadSize 512
#define NumberOfSignatures 10000000
#define MaxSignatureLength 16
#define MatchingIndex 99152
#define CopyLen 14

#define midPoint 5000000

const int blockSize = 64; 
const int gridSize = (midPoint + blockSize - 1) / blockSize;


#define match 2
#define mismatch -1
#define indel -1

#define matchmismatch 1
#define indelmismatch -2

__constant__ char PktPayload_d_constant[PayloadSize];

#endif

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

    int64_t row_size = (MaxSignatureLength+1)*midPoint;


    int32_t global_thread_id = block_id * block_size + thread_id;
    
    if(global_thread_id > midPoint){
        return;
    }

    int32_t signatures_offset = (global_thread_id) * (MaxSignatureLength + 1)-1;

    int32_t signatures_offset_1 = (global_thread_id + midPoint) * (MaxSignatureLength + 1)-1;

    // char thread_signature[MaxSignatureLength+1];
    // char thread_signature_1[MaxSignatureLength+1];
    
    
    for(int j =1; j< MaxSignatureLength+1;j++){
        // thread_signature[j] = signatures[signatures_offset + j];
        // thread_signature_1[j] = signatures[signatures_offset_1 + j];
        
        traceMatrix[global_thread_id + row_size] = 
        match * (PktPayload_d_constant[0] == signatures[signatures_offset + j])
        + ((match<<16) * (PktPayload_d_constant[0] ==  signatures[signatures_offset_1 + j]));
        
        
    }
    
    
    // uint32_t row1[MaxSignatureLength+1];
    // uint32_t row2[MaxSignatureLength+1];
    
    // for(int j =0; j< MaxSignatureLength+1;j++){
    //     row1[j] = traceMatrix[global_thread_id + row_size +(j)*midPoint];
    //     row2[j] = traceMatrix[global_thread_id + row_size +(j)*midPoint + row_size];
    // }
    

    uint32_t north, north_west, west,temp_hi, temp_low;

    bool even = 0;

    uint32_t current_value;

    for(int i = 2; i < PayloadSize+1; i ++){

        uint32_t iter_index = global_thread_id + row_size;
        
        even = !even;

        for(int j = 1; j < MaxSignatureLength+1; j ++){

            iter_index += midPoint;

            if(even){
                north_west = traceMatrix[global_thread_id + row_size +(j-1)*midPoint];
                north = traceMatrix[global_thread_id + row_size +(j)*midPoint];
                west = traceMatrix[global_thread_id + row_size +(j-1)*midPoint + row_size];
            }
            else {
                north_west = traceMatrix[global_thread_id + row_size +(j-1)*midPoint + row_size];
                north = traceMatrix[global_thread_id + row_size +(j)*midPoint + row_size];
                west = traceMatrix[global_thread_id + row_size +(j-1)*midPoint];
            }
            
            current_value = __vimax3_s16x2_relu(north,north_west,west);

            temp_hi = current_value &(0x0000FFFF);
            temp_low = current_value>>16;

            bool a =  PktPayload_d_constant[i-1] == signatures[signatures_offset + j];
            bool b =  PktPayload_d_constant[i-1] == signatures[signatures_offset_1 + j];

            current_value = (temp_hi + (matchmismatch) * (a) + (indelmismatch)* (!a)*(temp_hi>1) ) 
            | ((temp_low + (matchmismatch)  * (b)+ (indelmismatch)* (!b)*(temp_low>1))<<16);


            if(even){
                traceMatrix[global_thread_id + row_size +(j)*midPoint + row_size] = current_value;
            }
            else {
                traceMatrix[global_thread_id + row_size +(j)*midPoint] = current_value;
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

    if(free_mem > sizeof(int16_t) * totalBytes* 3) {

        printf("Tracing matrix size is %lld MB\n",traceMatrixSize/1000000);

        printf("Cuda memory available: %lld bytes\n", free_mem);

        cudaMemcpy(Signatures_d, Signatures, sizeof(char) * totalBytes, cudaMemcpyHostToDevice);
        err = cudaMalloc((void**)&traceMatrix_d, traceMatrixSize);
        CudaAllocErrorCheck;
        cudaMemcpy(traceMatrix_d, traceMatrix, traceMatrixSize, cudaMemcpyHostToDevice);
        cudaMemcpy(NumberOfSignatures_d, NumberOfSignatures_h, sizeof(u_int32_t), cudaMemcpyHostToDevice);

        // for(int i =0; i< 20;i++){
        //     printf("Warm up iteration %d\n",i);
        //      construct_trace_matrix_gpu_low<<<dim3(gridSize +1,1,1),dim3(blockSize,1,1)>>>(traceMatrix_d, Signatures_d,report_d);
        // }
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

    free(PktPayload);
    free(Signatures);

    cudaFree(PktPayload_d);
    cudaFree(Signatures_d);
    cudaFree(traceMatrix_d);

    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);

    return 0;
}