// Name:Reagan Burleson
// Vector addition on the GPU of any size with fixed block and grid size also adding pragma unroll for speed up.
// nvcc HW4.cu -o temp
/*
 What to do:
 This is the solution to HW3. It works well for adding vectors with fixed-size blocks. 
 Given the size of the vector it needs to add, it takes a set block size, determines how 
 many blocks are needed, and creates a grid large enough to complete the task. Cool, cool!
 
 But—and this is a big but—this can get you into trouble because there is a limited number 
 of blocks you can use. Though large, it is still finite. Therefore, we need to write the 
 code in such a way that we don't have to worry about this limit. Additionally, some block 
 and grid sizes work better than others, which we will explore when we look at the 
 streaming multiprocessors.
 
 Extend this code so that, given a block size and a grid size, it can handle any vector addition. 
 Start by hard-coding the block size to 256 and the grid size to 64. Then, experiment with different 
 block and grid sizes to see if you can achieve any speedup. Set the vector size to a very large value 
 for time testing.

 You’ve probably already noticed that the GPU doesn’t significantly outperform the CPU. This is because 
 we’re not asking the GPU to do much work, and the overhead of setting up the GPU eliminates much of the 
 potential speedup. 
 
 To address this, modify the computation so that:
 c = sqrt(cos(a)*cos(a) + a*a + sin(a)*sin(a) - 1.0) + sqrt(cos(b)*cos(b) + b*b + sin(b)*sin(b) - 1.0)
 Hopefully, this is just a convoluted and computationally expensive way to calculate a + b.
 If the compiler doesn't recognize the simplification and optimize away all the unnecessary work, 
 this should create enough computational workload for the GPU to outperform the CPU.

 Write the loop as a for loop rather than a while loop. This will allow you to also use #pragma unroll 
 to explore whether it provides any speedup. Make sure to include an if (id < n) condition in your code 
 to ensure safety. Finally, be prepared to discuss the impact of #pragma unroll and whether it helped 
 improve performance.
*/

// Include files
#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

// Defines
#define N 200000000      // Vector length 
#define BLOCK_SIZE 256   
#define GRID_SIZE  64    

// Global variables
float *A_CPU, *B_CPU, *C_CPU; //CPU pointers
float *A_GPU, *B_GPU, *C_GPU; //GPU pointers
dim3 BlockSize; //This variable will hold the Dimensions of your blocks
dim3 GridSize; //This variable will hold the Dimensions of your grid
float Tolerance = 0.01;

// Function prototypes
void cudaErrorCheck(const char *, int);
void setUpDevices();
void allocateMemory();
void innitialize();
void addVectorsCPU(float*, float*, float*, int);
__global__ void addVectorsGPU(float, float, float, int);
bool  check(float*, int);
long elaspedTime(struct timeval, struct timeval);
void cleanUp();

// Error checking 
#define cuda_check(stmt)                                                     \
  do {                                                                       \
    cudaError_t err__ = (stmt);                                              \
    if (err__ != cudaSuccess) {                                              \
      fprintf(stderr, "CUDA error: %s at %s:%d\n",                           \
              cudaGetErrorString(err__), __FILE__, __LINE__);                \
      exit(EXIT_FAILURE);                                                    \
    }                                                                        \
  } while (0)

// This will be the layout of the parallel space we will be using.
void setUpDevices()
{
	BlockSize = dim3(BLOCK_SIZE, 1, 1);
    GridSize  = dim3(GRID_SIZE, 1, 1);
}

// Allocating the memory we will be using.
void allocateMemory()
{	
	// Host "CPU" memory.				
	A_CPU = (float*)malloc(N*sizeof(float));
	B_CPU = (float*)malloc(N*sizeof(float));
	C_CPU = (float*)malloc(N*sizeof(float));

	if (!A_CPU || !B_CPU || !C_CPU) {
        printf("Host malloc failed\n");
        exit(1);
    }
	
	// Device "GPU" Memory
	cuda_check(cudaMalloc(&A_GPU, N * sizeof(float)));
    cuda_check(cudaMalloc(&B_GPU, N * sizeof(float)));
    cuda_check(cudaMalloc(&C_GPU, N * sizeof(float)));
}

// Loading values into the vectors that we will add.
void innitialize()
{
	for(int i = 0; i < N; i++)
	{		
		A_CPU[i] = (float)i;	
		B_CPU[i] = (float)(2*i);
	}
}

// heavy version of a+b
static inline float heavy_op(float a, float b) {
    float ca = cosf(a), sa = sinf(a);
    float cb = cosf(b), sb = sinf(b);
    float ta = sqrtf(ca*ca + a*a + sa*sa - 1.0f);
    float tb = sqrtf(cb*cb + b*b + sb*sb - 1.0f);
    return ta + tb;
}

// Adding vectors a and b on the CPU then stores result in vector c.
void addVectorsCPU(float *a, float *b, float *c, int n)
{
	for (int i = 0; i < n; i++) {
        c[i] = heavy_op(a[i], b[i]);
    }
}

// This is the kernel. It is the function that will run on the GPU.
// It adds vectors a and b on the GPU then stores result in vector c.
__global__ void addVectorsGPU(float *a, float *b, float *c, int n)
{
	int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    #pragma unroll 4
    for (int i = tid; i < n; i += stride) {
        float av = a[i];
        float bv = b[i];

        float ca = __cosf(av), sa = __sinf(av);
        float cb = __cosf(bv), sb = __sinf(bv);

        float ta = sqrtf(ca*ca + av*av + sa*sa - 1.0f);
        float tb = sqrtf(cb*cb + bv*bv + sb*sb - 1.0f);

        c[i] = ta + tb;
    }
}

// Checking to see if anything went wrong in the vector addition.
bool check(float *c, int n, float tolerence)
{
	int id;
	double myAnswer;
	double trueAnswer;
	double percentError;
	double m = n-1; // Needed the -1 because we start at 0.
	
	myAnswer = 0.0;
	for(id = 0; id < n; id++)
	{ 
		myAnswer += c[id];
	}
	
	trueAnswer = 3.0*(m*(m+1))/2.0;
	
	percentError = abs((myAnswer - trueAnswer)/trueAnswer)*100.0;
	
	if(percentError < Tolerance) 
	{
		return(true);
	}
	else 
	{
		return(false);
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
void CleanUp() {
    free(A_CPU); free(B_CPU); free(C_CPU);
    cuda_check(cudaFree(A_GPU));//changed to my cudacheck function
    cuda_check(cudaFree(B_GPU));
    cuda_check(cudaFree(C_GPU));
}

int main()
{
	timeval start, end;
	long timeCPU, timeGPU;
	
	// Setting up the GPU
	setUpDevices();
	
	// Allocating the memory you will need.
	allocateMemory();
	
	// Putting values in the vectors.
	innitialize();
	
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
	cuda_check(cudaMemcpyAsync(A_GPU, A_CPU, N*sizeof(float), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpyAsync(B_GPU, B_CPU, N*sizeof(float), cudaMemcpyHostToDevice));
	
	addVectorsGPU<<<GridSize,BlockSize>>>(A_GPU, B_GPU ,C_GPU, N);
	cuda_check(cudaGetLastError());       // check launch
    cuda_check(cudaDeviceSynchronize());  // wait for kernel
	
	// Copy Memory from GPU to CPU	
	cuda_check(cudaMemcpyAsync(C_CPU, C_GPU, N*sizeof(float), cudaMemcpyDeviceToHost));
    cuda_check(cudaDeviceSynchronize()); //making sure everything waited
	
	gettimeofday(&end, NULL);
	timeGPU = elaspedTime(start, end);
	
	// Checking to see if all went correctly.
	if(check(C_CPU, N, Tolerance) == false)
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
	printf("\n\n");
	
	return(0);
}
