#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>


#define Ver 12000

const int V = Ver;
  
const int INF = 99999;

#define blockSize 256
// int gridSize = (V*V + blockSize - 1) / blockSize;
int gridSize = 2048;
// 0.00003662109375
// 0.000017


__global__
void FloydWarshall_gpu(const int V, const int k, int *dis)
{   	 
    // int index2 = k*V;
    for (int i=blockIdx.x;i<V;i+=gridDim.x)
    {
		int temp = dis[i*V+k];
        bool val;
        int result;
		  for(int j=threadIdx.x;j<V;j+=blockDim.x)
		  {
                result = __vibmin_s32(temp + dis[k*V+j], dis[i*V+j],&val);
                if(val){
                    dis[i*V+j] = result;
                }

                // if(temp + dis[k*V +j] < dis[i*V+j] ){
                //     dis[i*V+j] = temp + dis[k*V +j];
                // }
            // }
		  }
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
