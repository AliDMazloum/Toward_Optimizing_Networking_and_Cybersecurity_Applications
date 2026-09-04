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


#define max(a,b) (((a)>(b))?(a):(b))
#define min(a,b) (((a)<(b))?(a):(b))

#define CHECK_CUDA_ERROR \
    do { \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err)); \
            return -1; \
        } \
    } while (0)

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

__global__ void construct_trace_matrix_gpu(int16_t *traceMatrix, const char* payload, const char* signatures,u_int32_t *report,
int* input_payloadSize, int* input_maxSignatureLength,int* input_numberOfSignatures){

    int thread_id = (blockDim.x*blockDim.y)*threadIdx.z+(threadIdx.y*blockDim.x)+(threadIdx.x);
    //BLock ID
    int block_id =(gridDim.x*gridDim.y)*blockIdx.z+(blockIdx.y*gridDim.x)+(blockIdx.x);
    //Block size
    int block_size = blockDim.x *blockDim.y*blockDim.z;
    //grid size

    int global_thread_id = block_id * block_size + thread_id;

    int16_t north, west, north_west;

    
    if(global_thread_id > *input_numberOfSignatures-1){
        return;
    }

    int64_t offset = global_thread_id * (*input_maxSignatureLength + 1)+1;
    // int64_t index;

    for(int i = 0; i< *input_maxSignatureLength+1; i++){
        traceMatrix[offset +i] = 0;
    }

    for(int64_t i = 0; i< *input_payloadSize; i++){
        traceMatrix[offset +i*(*input_maxSignatureLength+1)**input_numberOfSignatures] = 0;
    }

    for(int64_t i = 1; i < *input_payloadSize+1; i ++){
        for(int64_t j = 1; j < *input_maxSignatureLength+1; j ++){


            north = traceMatrix[offset + (i-1)*(*input_maxSignatureLength+1)**input_numberOfSignatures +j];
            west = traceMatrix[offset + i*(*input_maxSignatureLength+1)**input_numberOfSignatures +j-1];
            north_west = traceMatrix[offset + (i-1)*(*input_maxSignatureLength+1)**input_numberOfSignatures +j-1];

            // traceMatrix[offset + i*(*input_maxSignatureLength+1)**input_numberOfSignatures +j] = max(max(max(north+(signatures[offset +j-2] != '*')*indel, west+indel), north_west+
            //                          (signatures[offset +j-2] != '*')*(signatures[offset +j-2] != '.')*
            //                          ((payload[i-1] == signatures[offset +j-2])*match + (payload[i-1] != signatures[offset +j-2])*mismatch)),0);

            traceMatrix[offset + i*(*input_maxSignatureLength+1)**input_numberOfSignatures +j] = max(max(max(north+(signatures[offset +j-2] != '*')*indel, west+indel), north_west+
                                     (signatures[offset +j-2] != '*')*(signatures[offset +j-2] != '.')*(signatures[offset +j-2] != '~')*
                                     ((payload[i-1] == signatures[offset +j-2])*match + (payload[i-1] != signatures[offset +j-2])*mismatch) +(signatures[offset +j-2] == '~')*
                                     mismatch*(!dev_isdigit(payload[i-1]))),0);
            
            if (traceMatrix[offset + i*(*input_maxSignatureLength+1)**input_numberOfSignatures +j]>=(*input_maxSignatureLength*0.7*match)){
                report[0] = traceMatrix[offset + i*(*input_maxSignatureLength+1)**input_numberOfSignatures +j];
                report[1] = global_thread_id;
            }

        }
    }

}

void draw_trace_matrix(char * Signatures, int sigCount, int singSize, char* PktPayload,int payloadSize,  int16_t* traceMatrix){
    for(int i =0; i< payloadSize+2;i++){
        for(int j =0; j< singSize*sigCount+2; j++){
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
                int16_t row = (i-1) *(singSize+1)*sigCount;
                int16_t coloumn = j;
                printf("%d\t",traceMatrix[row +coloumn]);
            }
            
        }
        printf("\n");
    }
}

void print_usage(char* prog_name) {
    printf("\nUsage: %s [options]\n", prog_name);
    printf("Options:\n");
    printf("  --p_size <int>    Payload Size (Default: %d)\n", 50); // Replace with your constants
    printf("  --s_count <int>   Number of Signatures\n");
    printf("  --s_len <int>     Maximum Signature Length\n");
    printf("  --m_idx <int>     Matching Index\n");
    printf("  --block <int, int, int>     Block Dimentions\n");
    printf("  --grid <int, int, int>     Grid Dimentions\n");
    printf("  --help            Show this help message\n\n");
    printf("  --verbose            Show execution details like processing time\n\n");
}

int main(int argc, char* argv[]) {
    
    //parse the test parameters (Payload Size, number of signatures, maximum signature size)
    int input_payloadSize = PayloadSize;
    int input_numberOfSignatures = NumberOfSignatures;
    int input_maxSignatureLength = MaxSignatureLength;
    int input_matchingIndex = MatchingIndex;
    int verbose = 0;

    int bx = 10, by = 1, bz = 1;
    int gx = 1, gy = 1, gz = 1;

    for (int i = 1; i < argc; i++) {
        
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        }

        else if (strcmp(argv[i], "--p_size") == 0 && i + 1 < argc) {
            input_payloadSize = atoi(argv[++i]);
        } 
        else if (strcmp(argv[i], "--s_count") == 0 && i + 1 < argc) {
            input_numberOfSignatures = atoi(argv[++i]);
        } 
        else if (strcmp(argv[i], "--s_len") == 0 && i + 1 < argc) {
            input_maxSignatureLength = atoi(argv[++i]);
        } 
        else if (strcmp(argv[i], "--m_idx") == 0 && i + 1 < argc) {
            input_matchingIndex = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--block") == 0 && i + 1 < argc) {
            bx = atoi(argv[++i]); by = atoi(argv[++i]); bz = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--grid") == 0 && i + 1 < argc) {
            gx = atoi(argv[++i]); gy = atoi(argv[++i]); gz = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--verbose") == 0 && i < argc) {
            verbose = 1;
        }
        else {
            printf("Warning: Unknown or incomplete argument: %s\n", argv[i]);
        }
    }

    if(input_matchingIndex > input_numberOfSignatures){
        fprintf(stderr, "Matching index should be less than the number of signatures\n");
        return -1;
    }

    dim3 block = dim3(bx, by, bz);
    dim3 grid = dim3(gx, gy, gz);

    printf("p_size %d, s_count %d, s_len %d, m_idx %d, block (%d, %d, %d), grid(%d, %d, %d) \n",
         input_payloadSize, input_numberOfSignatures, input_maxSignatureLength, input_matchingIndex, bx, by, bz, gx, gy, gz );
        
    srand(time(NULL));
        
    size_t signaturesTotalBytes = input_numberOfSignatures * (input_maxSignatureLength+1) * sizeof(char);
    char* Signatures = (char*)malloc(signaturesTotalBytes);

    if (!Signatures) {
        fprintf(stderr, "Error: failed to allocate memory\n");
        exit(1);
    }

    for(int i =0; i< input_numberOfSignatures; i++){
        char* signature = generate_string(input_maxSignatureLength);
        strcpy(Signatures+i*(input_maxSignatureLength+1),signature);
        free(signature);
    }

    char *PktPayload = generate_string(input_payloadSize);

    char payloadPattern[] = "goosleMalicious.c0m";
    char singatureRegex[] = "goo.le*c~m";


    
    memcpy(PktPayload +5, payloadPattern, strlen(payloadPattern));
    memcpy(Signatures+input_matchingIndex*(input_maxSignatureLength+1), singatureRegex,strlen(singatureRegex));

    
    // printf("PktPayload is %s \n", PktPayload);
    // for(int i =0; i < input_numberOfSignatures; i++){
    //     printf("Signature[%d] is %s \n", i, &Signatures[i*(input_maxSignatureLength + 1)]);
    // }

    // CUDA CODE

    cudaError_t err;

    char *PktPayload_d;
    char *Signatures_d;

    size_t traceMatrixSize = sizeof(int16_t) * signaturesTotalBytes* (input_payloadSize+1);
    int16_t *traceMatrix = (int16_t*)calloc(1,traceMatrixSize);
    int16_t *traceMatrix_d;

    size_t reportSize = sizeof(u_int32_t)*2;
    u_int32_t *report = (u_int32_t*)calloc(2,sizeof(u_int32_t));
    u_int32_t *report_d;

    err = cudaMalloc((void**)&PktPayload_d, sizeof(char) * input_payloadSize);
    CHECK_CUDA_ERROR;

    err = cudaMalloc((void**)&Signatures_d, sizeof(char) * signaturesTotalBytes);
    CHECK_CUDA_ERROR;

    err = cudaMalloc((void**)&report_d, sizeof(u_int32_t) * 2);
    CHECK_CUDA_ERROR;

    err = cudaMalloc((void**)&traceMatrix_d, traceMatrixSize);
    CHECK_CUDA_ERROR;

    size_t free_mem, total_mem;

    err = cudaMemGetInfo(&free_mem, &total_mem);
    CHECK_CUDA_ERROR;
    if(verbose > 0){
        printf("Used memory:  %zu bytes (%.2f MB)\n", (total_mem - free_mem), (total_mem - free_mem) / (1024.0 * 1024));
        printf("Free memory: %zu bytes (%.2f MB)\n", free_mem, free_mem / (1024.0 * 1024));
    }


    int * input_payloadSize_d;
    int * input_maxSignatureLength_d;
    int * input_numberOfSignatures_d;

    err = cudaMalloc((void**)&input_payloadSize_d, sizeof(int));
    CHECK_CUDA_ERROR;

    err = cudaMalloc((void**)&input_maxSignatureLength_d, sizeof(int));
    CHECK_CUDA_ERROR;

    err = cudaMalloc((void**)&input_numberOfSignatures_d, sizeof(int));
    CHECK_CUDA_ERROR;

    cudaMemcpy(input_payloadSize_d, &input_payloadSize, sizeof(int) , cudaMemcpyHostToDevice);
    cudaMemcpy(input_maxSignatureLength_d, &input_maxSignatureLength, sizeof(int) , cudaMemcpyHostToDevice);
    cudaMemcpy(input_numberOfSignatures_d, &input_numberOfSignatures, sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(PktPayload_d, PktPayload, sizeof(char) * input_payloadSize, cudaMemcpyHostToDevice);
    cudaMemcpy(Signatures_d, Signatures, sizeof(char) * signaturesTotalBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(traceMatrix_d, traceMatrix, traceMatrixSize, cudaMemcpyHostToDevice);
    cudaMemcpy(report_d, report, reportSize, cudaMemcpyHostToDevice);



    clock_t start_clock, end_clock;

    start_clock = clock();
    
    construct_trace_matrix_gpu<<<grid,block>>>(traceMatrix_d, PktPayload_d, Signatures_d,report_d, input_payloadSize_d,
    input_maxSignatureLength_d, input_numberOfSignatures_d);

    // cudaMemcpy(traceMatrix, traceMatrix_d, traceMatrixSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(report, report_d, reportSize, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    end_clock= clock();
    printf("time taken by the GPU is %.6f\n",((double)end_clock - (double)start_clock)/CLOCKS_PER_SEC);

    // draw_trace_matrix(Signatures, input_numberOfSignatures, input_maxSignatureLength, PktPayload, input_payloadSize, traceMatrix);

    if(report[0] > 0){
        printf("Signature matching occured on signature %u with %u matching score\n",report[1],report[0]);
        // printf("Payload: %s\n", PktPayload);
        printf("Signature[%u]: %s \n", report[1],Signatures + report[1]*(input_maxSignatureLength + 1));
    }
    else{
        printf("No matching has been detecting\n");
    }

    free(PktPayload);
    free(Signatures);
    free(report);

    cudaFree(PktPayload_d);
    cudaFree(Signatures_d);
    cudaFree(traceMatrix_d);
    cudaFree(report_d);

    return 0;
}