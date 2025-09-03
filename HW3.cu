// Name: Reagan Burleson
// nvcc HW3.cu -o temp
/*
 What to do:
 This is the solution to HW2. It works well for adding vectors using a single block.
 But why use just one block?
 We have thousands of CUDA cores, so we should use many blocks to keep the SMs (Streaming Multiprocessors) on the GPU busy.

 Extend this code so that, given a block size, it will set the grid size to handle "almost" any vector addition.
 I say "almost" because there is a limit to how many blocks you can use, but this number is very large. 
 We will address this limitation in the next HW.

 Hard-code the block size to be 256.

 Also add cuda error checking into the code.
*/

// Include files
#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

// Defines
#define N 11503 // Length of the vector
#define B 256 //block size

// Error checking 
#define CUDA_CHECK(stmt)                                                     \
  do {                                                                       \
    cudaError_t err__ = (stmt);                                              \
    if (err__ != cudaSuccess) {                                              \
      fprintf(stderr, "CUDA error: %s at %s:%d\n",                           \
              cudaGetErrorString(err__), __FILE__, __LINE__);                \
      exit(EXIT_FAILURE);                                                    \
    }                                                                        \
  } while (0)

// Global variables
float *A_CPU, *B_CPU, *C_CPU; //CPU pointers
float *A_GPU, *B_GPU, *C_GPU; //GPU pointers
dim3 BlockSize; //This variable will hold the Dimensions of your blocks
dim3 GridSize; //This variable will hold the Dimensions of your grid
float Tolerance = 0.00000001;

// Function prototypes
void setUpDevices(int n);
void allocateMemory(int n);
void innitialize(int n);
void addVectorsCPU(float*, float*, float*, int);
__global__ void addVectorsGPU(float, float, float, int);
int  check(float*, int);
long elaspedTime(struct timeval, struct timeval);
void cleanUp(void);

// This will be the layout of the parallel space we will be using.
void setUpDevices(int n)
{
	BlockSize = dim3(B, 1, 1);

	// Compute num of blocks to cover n elements
	int blocksNeeded = (n + B - 1) / B;

	// GPUs hard limit for # of blocks
	int device = 0;
	cudaDeviceProp prop;
	CUDA_CHECK(cudaGetDevice(&device));
	CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
	int maxGridX = prop.maxGridSize[0];

	//decides how many blocks are launched, if blocksneeded is larger than allow we clamp it to maxgrid
	GridSize = dim3((unsigned) (blocksNeeded > maxGridX ? maxGridX : blocksNeeded), 1, 1);
}

// Allocating the memory we will be using.
void allocateMemory(int n)
{	
	//calculates how many bytes of mem are needed
	size_t bytes = (size_t)n * sizeof(float);

	// CPU			
	A_CPU = (float*)malloc(n*sizeof(float));
	B_CPU = (float*)malloc(n*sizeof(float));
	C_CPU = (float*)malloc(n*sizeof(float));
	
	//in case host does not have enough RAM
	if (!A_CPU || !B_CPU || !C_CPU) {
    fprintf(stderr, "Host malloc failed\n");
    exit(EXIT_FAILURE);
  }
	
	// GPU
	CUDA_CHECK(cudaMalloc((void**)&A_GPU, bytes));
	CUDA_CHECK(cudaMalloc((void**)&B_GPU, bytes));
	CUDA_CHECK(cudaMalloc((void**)&C_GPU, bytes));

}

// Loading values into the vectors that we will add.
void innitialize(int n)
{
	for(int i = 0; i < n; i++)
	{		
		A_CPU[i] = (float)i;	
		B_CPU[i] = (float)(2*i);
	}
}

// Adding vectors a and b on the CPU then stores result in vector c.
void addVectorsCPU(float *a, float *b, float *c, int n)
{
	for(int id = 0; id < n; id++)
	{ 
		c[id] = a[id] + b[id];
	}
}

// This is the kernel. It is the function that will run on the GPU.
// It adds vectors a and b on the GPU then stores result in vector c.
__global__ void addVectorsGPU(float *a, float *b, float *c, int n)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = id; i < n; i += stride) {
		c[i] = a[i] + b[i];
	}
}

// Checking to see if anything went wrong in the vector addition.
int check(float *c, int n)
{
	double sum = 0.0;
	double m = n-1; // Needed the -1 because we start at 0.
	
	for(int id = 0; id < n; id++)
	{ 
		sum += c[id];
	}
	
	if(abs(sum - 3.0*(m*(m+1))/2.0) < Tolerance) 
	{
		return(1);
	}
	else 
	{
		return(0);
	}
}

// Calculating elasped time.
long elaspedTime(struct timeval start, struct timeval end)
{
	// tv_sec = number of seconds past the Unix epoch 01/01/1970
	// tv_usec = number of microseconds past the current second.
	
	long startTime = start.tv_sec * 1000000 + start.tv_usec; // In microseconds.
	long endTime = end.tv_sec * 1000000 + end.tv_usec; // In microseconds

	// Returning the total time elasped in microseconds
	return endTime - startTime;
}

// Cleaning up memory after we are finished.
void CleanUp(void)
{
	// Freeing host "CPU" memory.
	if (A_CPU) free(A_CPU);
	if (B_CPU) free(B_CPU);
	if (C_CPU) free(C_CPU);

	if (A_GPU) CUDA_CHECK(cudaFree(A_GPU));
	if (B_GPU) CUDA_CHECK(cudaFree(B_GPU));
	if (C_GPU) CUDA_CHECK(cudaFree(C_GPU));
}

int main()
{
	timeval start, end;
	long timeCPU, timeGPU;
	
	// Setting up the GPU
	setUpDevices(N);
	
	// Allocating the memory you will need.
	allocateMemory(N);
	
	// Putting values in the vectors.
	innitialize(N);
	
	// Adding on the CPU
	gettimeofday(&start, NULL);
	addVectorsCPU(A_CPU, B_CPU ,C_CPU, N);
	gettimeofday(&end, NULL);
	timeCPU = elaspedTime(start, end);
	
	// Zeroing out the C_CPU vector just to be safe because right now it has the correct answer in it.
	for(int id = 0; id < N; id++)
	{ 
		C_CPU[id] = 0.0;
	}
	
	// Adding on the GPU
	gettimeofday(&start, NULL);
	
	// Copy Memory from CPU to GPU		
	size_t bytes = (size_t)N * sizeof(float);
	CUDA_CHECK(cudaMemcpyAsync(A_GPU, A_CPU, bytes, cudaMemcpyHostToDevice));
  	CUDA_CHECK(cudaMemcpyAsync(B_GPU, B_CPU, bytes, cudaMemcpyHostToDevice));
	
	addVectorsGPU<<<GridSize,BlockSize>>>(A_GPU, B_GPU ,C_GPU, N);
	// Catch launch errors early
  	CUDA_CHECK(cudaGetLastError());
	
	// Copy Memory from GPU to CPU	
	CUDA_CHECK(cudaMemcpyAsync(C_CPU, C_GPU, bytes, cudaMemcpyDeviceToHost));

	// Making sure the GPU and CPU wiat until each other are at the same place.
	CUDA_CHECK(cudaDeviceSynchronize());
	
	gettimeofday(&end, NULL);
	timeGPU = elaspedTime(start, end);
	
	// Checking to see if all went correctly.
	if(check(C_CPU, N) == 0)
	{
		printf("\n\n Something went wrong in the GPU vector addition\n");
	}
	else
	{
		printf("\n\n You added the two vectors correctly on the GPU");
		printf("\n The time it took on the CPU was %ld microseconds", timeCPU);
		printf("\n The time it took on the GPU was %ld microseconds", timeGPU);
	}
	
	// Your done so cleanup your room.	
	CleanUp();	
	
	// Making sure it flushes out anything in the print buffer.
	printf("\n");
	
	return(0);
}

