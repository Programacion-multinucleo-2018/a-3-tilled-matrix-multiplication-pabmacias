#include "common.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

using namespace std;

#define SIZE 1000
#define TILE 32

void initialData(float *ip, const int size)
{
    int i;

    for(i = 0; i < size; i++)
    {
        ip[i] = (float)(rand() & 0xFF);
    }

    return;
}

__global__ void multMatrixOnGPU2D(float *MatA, float *MatB, float *MatC, int nx,
    int ny)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * nx + ix;

    if (ix < nx && iy < ny)
      for (int im = 0; im < ny; im++)
      {
        int idxm = iy * nx + im;
        int idym = im * nx + ix;
        MatC[idx] += MatA[idxm] * MatB[idym];
      }
}

__global__ void multMatrixOnGPUTiles(float *MatA, float *MatB, float *MatC, int nx,
    int ny)
{
    __shared__ float tile_x[TILE][TILE];
    __shared__ float tile_y[TILE][TILE];

    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;

    float result = 0;
    int max_tile = (TILE+SIZE-1) / TILE;

  	for (int i=max_tile; i>=0; i--)
  	{
  		if (i*TILE+threadIdx.x < SIZE && iy<SIZE) {
  			tile_x[threadIdx.y][threadIdx.x] = MatA[iy*SIZE + i*TILE + threadIdx.x];
  		}

  		if (i*TILE+threadIdx.y < SIZE && ix<SIZE) {
  			tile_y[threadIdx.y][threadIdx.x] = MatB[(i*TILE + threadIdx.y) * SIZE+ix];
  		}

  		for (int j=0; j<TILE; j++) {
  			result += tile_x[threadIdx.y][j] * tile_y[j][threadIdx.x];
  		}
  		__syncthreads();
    }

  	if (ix < SIZE && iy < SIZE) {
  		MatC[iy * SIZE + ix] = result;
  	}
}

int main(int argc, char **argv)
{
    printf("%s Starting...\n", argv[0]);

    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    SAFE_CALL(cudaGetDeviceProperties(&deviceProp, dev), "Error device prop");
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    SAFE_CALL(cudaSetDevice(dev), "Error setting device");

    // set up data size of matrix
    int nx = SIZE;
    int ny = SIZE;

    int nxy = nx * ny;
    int nBytes = nxy * sizeof(float);
    printf("Matrix size: nx %d ny %d\n", nx, ny);

    // malloc host memory
    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    hostRef = (float *)malloc(nBytes);
    gpuRef = (float *)malloc(nBytes);

    // initialize data at host side

    initialData(h_A, nxy);
    initialData(h_B, nxy);

    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    /* Deje comentada esta parte porque estaba tardando mucho en cpu para hacer
    las pruebas, pero si lo cheque antes y los resultados estaban bien */

    // add matrix at host side for result SAFE_CALLs
    /*auto start_cpu =  chrono::high_resolution_clock::now();
    multMatrixOnHost(h_A, h_B, hostRef, nx, ny);
    auto end_cpu =  chrono::high_resolution_clock::now();
    chrono::duration<float, std::milli> duration_ms = end_cpu - start_cpu;

    printf("multMatrixOnHost elapsed %f ms\n", duration_ms.count());*/

    // malloc device global memory
    float *d_MatA, *d_MatB, *d_MatC;
    SAFE_CALL(cudaMalloc((void **)&d_MatA, nBytes), "Error allocating d_MatA");
    SAFE_CALL(cudaMalloc((void **)&d_MatB, nBytes), "Error allocating d_MatB");
    SAFE_CALL(cudaMalloc((void **)&d_MatC, nBytes), "Error allocating d_MatC");

    // transfer data from host to device
    SAFE_CALL(cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice), "Error copying d_MatA");
    SAFE_CALL(cudaMemcpy(d_MatB, h_B, nBytes, cudaMemcpyHostToDevice), "Error copying d_MatB");

    // invoke kernel at host side
    dim3 block(TILE, TILE);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    auto start_cpu =  chrono::high_resolution_clock::now();
    multMatrixOnGPUTiles<<<grid, block>>>(d_MatA, d_MatB, d_MatC, nx, ny);
    SAFE_CALL(cudaDeviceSynchronize(), "Error executing kernel");
    auto end_cpu =  chrono::high_resolution_clock::now();

    chrono::duration<float, std::milli> duration_ms = end_cpu - start_cpu;

    printf("multMatrixOnGPUTiles <<<(%d,%d), (%d,%d)>>> elapsed %f ms\n", grid.x,
           grid.y,
           block.x, block.y, duration_ms.count());

   float *d_MatD;
   SAFE_CALL(cudaMalloc((void **)&d_MatD, nBytes), "Error allocating d_MatD");

   start_cpu =  chrono::high_resolution_clock::now();
   multMatrixOnGPU2D<<<grid, block>>>(d_MatA, d_MatB, d_MatD, nx, ny);
   SAFE_CALL(cudaDeviceSynchronize(), "Error executing kernel");
   end_cpu =  chrono::high_resolution_clock::now();

   duration_ms = end_cpu - start_cpu;

   printf("multMatrixOnGPU2D <<<(%d,%d), (%d,%d)>>> elapsed %f ms\n", grid.x,
          grid.y,
          block.x, block.y, duration_ms.count());

    // SAFE_CALL kernel error
    SAFE_CALL(cudaGetLastError(), "Error with last error");

    // copy kernel result back to host side
    SAFE_CALL(cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost), "Error copying d_MatC");

    // free device global memory
    SAFE_CALL(cudaFree(d_MatA), "Error freeing memory");
    SAFE_CALL(cudaFree(d_MatB), "Error freeing memory");
    SAFE_CALL(cudaFree(d_MatC), "Error freeing memory");

    // free host memory
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    // reset device
    SAFE_CALL(cudaDeviceReset(), "Error reseting");

    return (0);
}
