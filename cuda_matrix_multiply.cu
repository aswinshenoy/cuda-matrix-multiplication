#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 16  // threads per block

void multiply_using_cpu(int *m1, int *m2, int *rcpu, int m, int n, int k) {
    for (int row = 0; row < m; row++) 
        for (int col = 0; col < k; col++) 
        {
            int sum = 0;
            for (int i = 0; i < n; i++) 
                sum += m1[row * n + i] * m2[i * k + col];
            rcpu[row * k + col] = sum;
        }
}

__global__ void multiply_using_gpu(int *m1, int *m2, int *rgpu, int m, int n, int k)
{ 
    // We no longer need to run 2 for loops over row & col, but rather
    // we directly find the row & column assigned to the thread in the resultant matrix - 
    // as each thread is assigned to a specific element.
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread in one specific block is identified by threadIdx.x and threadIdx.y. 
    // Each block is one specific grid is identified by blockIdx.x and blockIdx.y. 
    // Therefore, if we have threadIdx.x, threadIdx.y, blockIdx.x and blockIdx.y, we can locate one specific thread.

    // Boundary protection - as there is chance of block having more threads than required
    // There is no problem if k & m are multiples of BLOCK_SIZE, but otherwise there is.
    if(col < k && row < m) 
    {
      int sum = 0;
      for(int i = 0; i < n; i++)
          // summing up & accumulating result for a single cell in resultant matrix  
          sum += m1[row * n + i] * m2[i * k + col];
      // write the result
      rgpu[row * k + col] = sum;
    }
}

void initialize_matrix(int *matrix, int m, int n) {
    //  We linearize the matrices to a 2D array and 
    //  stack each row lengthways, from the first to the last.  (Row Major Ordering)
    for (int row = 0; row < m; row++)
        for (int col = 0; col < n; col++)
            matrix[row * n + col] = rand();  
}

int check_for_discrepancies(int *r1, int *r2, int m, int n, int k) {
    int isValidated = 1;
    for (int i = 0; i < m; i++)
        for (int j = 0; j < k; j++)
            if(r1[i*k + j] != r2[i*k + j])
                isValidated = 0;
    return isValidated;
}

float test_case(int m, int n, int k) {
    printf("Multiplying a %d x %d matrix with %d x %d matrix \n", m, n, n, k);
    printf("---------------------\n\n");
    
    // allocate memory space for matrices & results on the host
    int *m1, *m2, *rcpu, *rgpu;
    cudaMallocHost((void **) &m1, sizeof(int)*m*n);  // Efficient way of writing - int *m1 = (int*)malloc(sizeof(int)*m*n); 
    cudaMallocHost((void **) &m2, sizeof(int)*n*k);
    cudaMallocHost((void **) &rcpu, sizeof(int)*m*k);
    cudaMallocHost((void **) &rgpu, sizeof(int)*m*k);

    initialize_matrix(m1, m, n);
    initialize_matrix(m2, n, k);

    float gpu_time_ms, cpu_time_ms;

    // some events to count the execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
 
    // start the CPU version
    cudaEventRecord(start, 0);

    multiply_using_cpu(m1, m2, rcpu, m, n, k);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cpu_time_ms, start, stop);
    printf("CPU: %f ms.\n", cpu_time_ms);

    // start to count execution time of GPU version
    cudaEventRecord(start, 0);
 
    // Allocate memory space on the device 
    int *dm1, *dm2, *dr;
    cudaMalloc((void **) &dm1, sizeof(int)*m*n);
    cudaMalloc((void **) &dm2, sizeof(int)*n*k);
    cudaMalloc((void **) &dr, sizeof(int)*m*k);

    // copy matrix A and B from host to device memory
    cudaMemcpy(dm1, m1, sizeof(int)*m*n, cudaMemcpyHostToDevice);
    cudaMemcpy(dm2, m2, sizeof(int)*n*k, cudaMemcpyHostToDevice);

    // We arrange the thread-blocks and grid in 2-D as we want to multiply a 2D matrix
    unsigned int grid_rows = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 grid(grid_cols, grid_rows);
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

    // The function is a CUDA kernel, and is executed by an array of CUDA threads. All threads run the same code. 
    multiply_using_gpu<<<grid,threads>>>(dm1, dm2, dr, m, n, k);    

    // Copy back the results from the device to host
    cudaMemcpy(rgpu, dr, sizeof(int)*m*k, cudaMemcpyDeviceToHost);
 
    cudaThreadSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time_ms, start, stop);
    printf("GPU: %f ms.\n", gpu_time_ms);

    // Freeup device memory
    cudaFree(dm1);
    cudaFree(dm2);
    cudaFree(dr);
 
    printf("\n");
 
    // Check for discrepencies & compute & show speedUp
    if(check_for_discrepancies(rcpu, rgpu, m, n, k))
        printf("Speedup = %f \n\n\n", cpu_time_ms / gpu_time_ms);
    else
        printf("Results from CPU & GPU are not matching. \n\n\n");


    // Freeup host memory
    cudaFreeHost(m1);
    cudaFreeHost(m2);
    cudaFreeHost(rcpu);
    cudaFreeHost(rgpu);

    return cpu_time_ms / gpu_time_ms;  // return speedup
}

int main()
{
    float s0, s1, s2, s3, s4;
    s0 = test_case(128, 64, 128);
    s1 = test_case(256, 128, 256);
    s2 = test_case(512, 256, 512);
    s3 = test_case(1024, 512, 1024);
    s4 = test_case(2048, 1024, 2048);
   
    printf("Summary of SpeedUps\n");
    printf("---------------------\n");
    printf("128 x 64 Matrix = %f\n", s0);
    printf("256 x 128 Matrix = %f\n", s1);
    printf("512 x 256 Matrix = %f\n", s2);
    printf("1024 x 512 Matrix = %f\n", s3);
    printf("2048 x 1024 Matrix = %f\n", s4);
 
    return 0;
}
}