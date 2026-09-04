#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <ctype.h>

#ifndef HELPERS_H
#define HELPERS_H

#define PayloadSize 50
#define NumberOfSignatures 10000
#define MaxSignatureLength 20

#define MatchingIndex 1356

#define match 6
#define mismatch -3
#define indel -2

// #define match 2
// #define mismatch -1
// #define indel -1

#define max(a,b) (((a)>(b))?(a):(b))
#define min(a,b) (((a)<(b))?(a):(b))

#endif

__device__ __forceinline__ bool dev_isdigit(char c) {
    unsigned char uc = (unsigned char)c;
    return (uc >= '0' && uc <= '9');
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

__global__ void construct_trace_matrix_gpu(int16_t *traceMatrix, const char* payload, const char* signatures,u_int32_t *report){

    int thread_id = (blockDim.x*blockDim.y)*threadIdx.z+(threadIdx.y*blockDim.x)+(threadIdx.x);
    //BLock ID
    int block_id =(gridDim.x*gridDim.y)*blockIdx.z+(blockIdx.y*gridDim.x)+(blockIdx.x);
    //Block size
    int block_size = blockDim.x *blockDim.y*blockDim.z;
    //grid size

    int global_thread_id = block_id * block_size + thread_id;

    int16_t north, west, north_west;

    
    if(global_thread_id > NumberOfSignatures-1){
        return;
    }

    int64_t offset = global_thread_id * (MaxSignatureLength + 1)+1;
    // int64_t index;

    for(int i = 0; i< MaxSignatureLength+1; i++){
        traceMatrix[offset +i] = 0;
    }

    for(int64_t i = 0; i< PayloadSize; i++){
        traceMatrix[offset +i*(MaxSignatureLength+1)*NumberOfSignatures] = 0;
    }

    for(int64_t i = 1; i < PayloadSize+1; i ++){
        for(int64_t j = 1; j < MaxSignatureLength+1; j ++){


            north = traceMatrix[offset + (i-1)*(MaxSignatureLength+1)*NumberOfSignatures +j];
            west = traceMatrix[offset + i*(MaxSignatureLength+1)*NumberOfSignatures +j-1];
            north_west = traceMatrix[offset + (i-1)*(MaxSignatureLength+1)*NumberOfSignatures +j-1];

            // traceMatrix[offset + i*(MaxSignatureLength+1)*NumberOfSignatures +j] = max(max(max(north+(signatures[offset +j-2] != '*')*indel, west+indel), north_west+
            //                          (signatures[offset +j-2] != '*')*(signatures[offset +j-2] != '.')*
            //                          ((payload[i-1] == signatures[offset +j-2])*match + (payload[i-1] != signatures[offset +j-2])*mismatch)),0);

            traceMatrix[offset + i*(MaxSignatureLength+1)*NumberOfSignatures +j] = max(max(max(north+(signatures[offset +j-2] != '*')*indel, west+indel), north_west+
                                     (signatures[offset +j-2] != '*')*(signatures[offset +j-2] != '.')*(signatures[offset +j-2] != '~')*
                                     ((payload[i-1] == signatures[offset +j-2])*match + (payload[i-1] != signatures[offset +j-2])*mismatch) +(signatures[offset +j-2] == '~')*
                                     mismatch*(!dev_isdigit(payload[i-1]))),0);
            
            if (traceMatrix[offset + i*(MaxSignatureLength+1)*NumberOfSignatures +j]>=40){
                report[0] = traceMatrix[offset + i*(MaxSignatureLength+1)*NumberOfSignatures +j];
                report[1] = global_thread_id;
            }

        }
    }

}

void draw_trace_matrix(char * Signatures, char* PktPayload, int16_t* traceMatrix){
    for(int i =0; i< PayloadSize+2;i++){
        for(int j =0; j< MaxSignatureLength*NumberOfSignatures+2; j++){
            if(i == 0){
                if(j <2){
                    printf("\t");
                }else{
                    printf("%c\t",Signatures[j-2]);
                }
            }
            else if(j == 0){
                if(i < 2){
                    printf("\t");
                }else{
                    printf("%c\t",PktPayload[i-2]);
                }
            }else{
                int16_t row = (i-1) *(MaxSignatureLength+1)*NumberOfSignatures;
                int16_t coloumn = j;
                printf("%d\t",traceMatrix[row +coloumn]);
            }
            
        }
        printf("\n");
    }
}

int main() {
    srand(time(NULL));

    FILE *f = fopen("signatures.txt","w");
    if(f){
        generate_signatures(f);
        fclose(f);
    }
    
    // char * PktPayload = generate_string(PayloadSize);
    
    // size_t totalBytes = NumberOfSignatures * MaxSignatureLength * sizeof(char) + NumberOfSignatures*sizeof(char);
    // char* Signatures = (char*)malloc(totalBytes);
    
    // if(!Signatures){
    //     fprintf(stderr, "Error: failed to allocate memory\n");
    //     exit(1);
    // }
    
    // f = fopen("signatures.txt","r");
    // char buffer[MaxSignatureLength +2];
    // int index = 0;
    // while(fgets(buffer,MaxSignatureLength,f)){
    //     strcpy(Signatures+index*(MaxSignatureLength+1),buffer);
    //     index+=1;
    //     printf("%d %zu\n",index,totalBytes);
    // }
    // fclose(f);

    size_t totalBytes = NumberOfSignatures * (MaxSignatureLength+1) * sizeof(char);
    char* Signatures = (char*)malloc(totalBytes);

    if (!Signatures) {
        fprintf(stderr, "Error: failed to allocate memory\n");
        exit(1);
    }

    char *PktPayload = generate_string(PayloadSize);
    // strncpy(PktPayload, "goo3leMalicious.c5461m", 19);
    
    
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
    
    // strncpy(PktPayload, Signatures+MatchingIndex*(MaxSignatureLength+1),35);
    // strncpy(Signatures+MatchingIndex*(MaxSignatureLength+1), "goosleMalicious.co9",19);
    strncpy(PktPayload, "goosleMalicious.c0m", 30);
    strncpy(Signatures+MatchingIndex*(MaxSignatureLength+1), "goo.le*c~m",10);
    fclose(f);
    
    // printf("%s\n",PktPayload);
    // printf("%.*s\n", 10, PktPayload);
    // printf("%.*s\n", 10, Signatures+MatchingIndex*(MaxSignatureLength+1));

    char *PktPayload_d;
    char *Signatures_d;

    int16_t *traceMatrix = (int16_t*)malloc(sizeof(int16_t) * totalBytes* (PayloadSize+1));
    if (traceMatrix) memset(traceMatrix, 0, sizeof(int16_t) * totalBytes* (PayloadSize+1));
    int16_t *traceMatrix_d;

    u_int32_t *report = (u_int32_t*)malloc(2*sizeof(u_int32_t));
    report[0] = 0;
    report[1] = 0;
    u_int32_t *report_d;

    cudaError_t err = cudaMalloc((void**)&PktPayload_d, sizeof(char) * PayloadSize);

    if (err != cudaSuccess) {
        // Allocation failed
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    err = cudaMalloc((void**)&Signatures_d, sizeof(char) * totalBytes);

    if (err != cudaSuccess) {
        // Allocation failed
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    err = cudaMalloc((void**)&report_d, sizeof(u_int32_t) * 2);

    if (err != cudaSuccess) {
        // Allocation failed
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    err = cudaMalloc((void**)&traceMatrix_d, sizeof(int16_t) * totalBytes* (PayloadSize+1));

    if (err != cudaSuccess) {
        // Allocation failed
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return -1;
    }else{
        printf("Tracing matrix size is %lld\n",sizeof(int16_t) * totalBytes* (PayloadSize+1));
    }

    cudaMemcpy(PktPayload_d, PktPayload, sizeof(char) * PayloadSize, cudaMemcpyHostToDevice);
    cudaMemcpy(Signatures_d, Signatures, sizeof(char) * totalBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(traceMatrix_d, traceMatrix, sizeof(int16_t) * totalBytes* (PayloadSize+1), cudaMemcpyHostToDevice);
    cudaMemcpy(report_d, report, 2*sizeof(u_int32_t), cudaMemcpyHostToDevice);

    clock_t start_clock, end_clock;

    start_clock = clock();

    construct_trace_matrix_gpu<<<dim3(10,1,1),dim3(10,10,10)>>>(traceMatrix_d, PktPayload_d, Signatures_d,report_d);
    // construct_trace_matrix_gpu<<<(NumberOfSignatures + 255)/256,256>>>(traceMatrix_d, PktPayload_d, Signatures_d,report_d);

    
    cudaMemcpy(traceMatrix, traceMatrix_d, sizeof(int16_t) * totalBytes* (PayloadSize+1), cudaMemcpyDeviceToHost);
    cudaMemcpy(report, report_d, 2*sizeof(u_int32_t), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    end_clock= clock();
    printf("time taken by the GPU is %.6f\n",((double)end_clock - (double)start_clock)/CLOCKS_PER_SEC);


    // draw_trace_matrix(Signatures, PktPayload, traceMatrix);

    if(report[0] > 0){
        printf("Signature matching occured on signature %u with %u matching score\n",report[1],report[0]);
    }
    else{
        printf("No matching has been detecting\n");
    }


    size_t free_mem, total_mem;

    err = cudaMemGetInfo(&free_mem, &total_mem);

    if (err != cudaSuccess) {
        printf("cudaMemGetInfo failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    printf("Free memory:  %zu bytes (%.2f MB)\n", free_mem, free_mem / (1024.0 * 1024));
    printf("Total memory: %zu bytes (%.2f MB)\n", total_mem, total_mem / (1024.0 * 1024));


    free(PktPayload);
    free(Signatures);

    cudaFree(PktPayload_d);
    cudaFree(Signatures_d);
    cudaFree(traceMatrix_d);

    return 0;
}