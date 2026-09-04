#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>


"""
This Code is used for the INDIS paper
"""

#define Ver 2000

const int V = Ver;
  
const int INF = 99999;

int blockSize = 256;
// int gridSize = (V*V /+ blockSize - 1) / blockSize;
int gridSize = 1024;

// Kernel function for inner two loop of Floyd Warshall Algorithm
__global__
void FloydWarshall_gpu(const int V, const int k, int *dis)
{
	// int t= (blockDim.x*blockDim.y)*threadIdx.z+(threadIdx.y*blockDim.x)+(threadIdx.x);
	// int b= (gridDim.x*gridDim.y)*blockIdx.z+(blockIdx.y*gridDim.x)+(blockIdx.x);
	// int T= blockDim.x*blockDim.y*blockDim.z;
	// int B= gridDim.x*gridDim.y*gridDim.z;

	// // int tm;

	// // __shared__ int coloumn[Ver];

	// // for (int i =0; i< V; i++){
	// // 	coloumn[i] = dis[k*V +i];
	// // }
	 
    // for (int i=b;i<V;i+=B)
    // {
	// 	int temp = dis[i*V+k];
	// 	  for(int j=t;j<V;j+=T)
	// 	  {
    //         // if(i*V+j < V*V){
    //             dis[i*V+j] = __viaddmin_s32(temp, dis[k*V +j], dis[i*V+j]);
	// 			// tm = dis[i*V+k] + dis[k*V+j];
	// 			// dis[i*V+j] = tm*(tm < dis[i*V+j])+ dis[i*V+j]*(tm >= dis[i*V+j]);
    //         // }
	// 	  }
	// }


	int thread_id = (blockDim.x*blockDim.y)*threadIdx.z+(threadIdx.y*blockDim.x)+(threadIdx.x);
    //BLock ID
    int block_id =(gridDim.x*gridDim.y)*blockIdx.z+(blockIdx.y*gridDim.x)+(blockIdx.x);
    //Block size
    int block_size = blockDim.x *blockDim.y*blockDim.z;
    //grid size

    int global_thread_id = block_id * block_size + thread_id;
	int total_number_of_threads = gridDim.x*gridDim.y*gridDim.z*block_size;

    int temp = k*V;

	for(int64_t i = global_thread_id; i < V*V; i +=total_number_of_threads){
		int row = i/V;
		int coloumn = i%V;
        // dis[row * V + coloumn] = __viaddmin_s32(dis[row*V+k], dis[k*V+coloumn], dis[row * V + coloumn]);
        // dis[i] = __viaddmin_s32(dis[row*V+k], dis[temp+coloumn], dis[i]);
        dis[i] = __viaddmin_s32(dis[block_id*V+k], dis[temp+thread_id], dis[i]);
        // printf("Thread %d processing element %lld (row %d, column %d)\n", global_thread_id, i, row, coloumn);
		// int row = i%V;
		// int coloumn = i/V;
        // dis[i] = __viaddmin_s32(dis[row*V+k], dis[k*V+coloumn], dis[i]);
    }
}

void FloydWarshall_cpu(int V, int *dis)
{
    int tm;
	
    for (int k = 0; k < V; k++)  
  	{
        for (int i=0;i<V;i++)
        {
            for(int j=0;j<V;j++)
            {
                tm = dis[i*V+k] + dis[k*V+j];
                dis[i*V+j] = tm*(tm < dis[i*V+j])+ dis[i*V+j]*(tm >= dis[i*V+j]);
    
                //   tm = dis[i*V+k] + dis[k*V+j];
                //   if(tm < dis[i*V+j] ){
                // 		dis[i*V+j] = tm;
                //   }
                // if(i*V+j < V*V){
                //     dis[i*V+j] = __viaddmin_s32(dis[i*V+k], dis[k*V+j], dis[i*V+j]);
                // }
            }
        }
        
	}
} 
  

int main(void)
{
	// int dev = 0;
    // cudaDeviceProp prop;
    // cudaGetDeviceProperties(&prop, dev);

    // printf("Device %d: %s\n", dev, prop.name);
    // printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
    // printf("  Total global memory: %zu MB\n", prop.totalGlobalMem / (1024 * 1024));
    // printf("  Multiprocessors (SMs): %d\n", prop.multiProcessorCount);
    // printf("  Max threads per multiprocessor: %d\n", prop.maxThreadsPerMultiProcessor);
    // printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
    // printf("  Max threads dimensions: (%d, %d, %d)\n",
    //         prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    // printf("  Max grid size: (%d, %d, %d)\n",
    //         prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    // printf("  Shared memory per block: %zu KB\n", prop.sharedMemPerBlock / 1024);
    // printf("  Warp size: %d\n", prop.warpSize);
    // printf("  Registers per block: %d\n", prop.regsPerBlock);
    // printf("  Clock rate: %.2f GHz\n", prop.clockRate / 1e6);
    // printf("  Memory Clock Rate: %.2f GHz\n", prop.memoryClockRate / 1e6);
    // printf("  Memory Bus Width: %d bits\n", prop.memoryBusWidth);
    // printf("\n");

	// nvmlInit();
	// nvmlDevice_t device;
	// nvmlDeviceGetHandleByIndex(0, &device);

	// unsigned int power_before; // in milliwatts
	// nvmlDeviceGetPowerUsage(device, &power_before);
	// nvmlShutdown();

	int *dis = (int*)malloc(V*V*sizeof(int));
	int *dis_d;

	cudaError_t err = cudaMalloc((void**)&dis_d, V*V*sizeof(int));

	if (err != cudaSuccess) {
        // Allocation failed
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return -1;
    }


	// initialize dis array on the host
	for (int i = 0; i < V; i++) 
	{
		for(int j = 0; j < V; j++)
		{
			
			if(j==i+1) dis[i*V+j] = 1;
			else if (i!=j) dis[i*V+j] = INF;
			else dis[i*V+j] = 0;
		}
	}

	
	cudaMemcpy(dis_d, dis, V*V*sizeof(int), cudaMemcpyHostToDevice);
    clock_t start_clock, end_clock;

    start_clock = clock();

    for (int k = 0; k < V; k++)  
  	{
        FloydWarshall_gpu<<<dim3(gridSize,1,1),dim3(blockSize,1,1)>>>(V, k, dis_d);
        // GPUInnerLoops<<<dim3(4,4,4), dim3(2,2,2)>>>(V, k, dis);
        cudaDeviceSynchronize();
	}

    end_clock= clock();
	cudaMemcpy(dis, dis_d, V*V*sizeof(int), cudaMemcpyDeviceToHost);
    printf("time taken by the GPU is %.6f\n",((double)end_clock - (double)start_clock)/CLOCKS_PER_SEC);

	// unsigned int power_after;
	// nvmlDeviceGetPowerUsage(device, &power_after);
	// printf("Power: %.2f, %.2f, %.2f W\n",power_after/ 1000.0,power_before/ 1000.0, (power_after - power_before) / 1000.0);
	// nvmlShutdown();

    // start_clock = clock();

    // FloydWarshall_cpu(V, dis);

    // end_clock= clock();
    // printf("time taken by the CPU is %.6f\n",((double)end_clock - (double)start_clock)/CLOCKS_PER_SEC);
	
	for (int i = 0; i < V; i++) 
	{
		for(int j = 0; j < V; j++)
		{
			if(j>=i) 
	 		{
		 		assert( dis[i*V+j] == j-i);
			}
			else assert( dis[i*V+j] == INF);
		}
	}

  // Free memory
	cudaFree(dis);
	return 0;
}
