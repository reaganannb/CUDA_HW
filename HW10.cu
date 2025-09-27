// Name: Reagan Burleson
// Robust Vector Dot product 
// nvcc HW10.cu -o temp
/*
 What to do:
 This code is the solution to HW9. It computes the dot product of vectors of any length and uses shared memory to 
 reduce the number of calls to global memory. However, because blocks can't sync, it must perform the final reduction 
 on the CPU. 
 To make this code a little less complicated on the GPU let do some pregame stuff and use atomic adds.
 1. Make sure the number of threads on a block are a power of 2 so we don't have to see if the fold is going to be
    even. Because if it is not even we had to add the last element to the first reduce the fold by 1 and then fold. 
    If it is not even tell your client what is wrong and exit.
 2. Find the right number of blocks to finish the job. But, it is possible that the grid demention is too big. I know
    it is a large number but it is finite. So use device properties to see if the grid is too big for the machine 
    you are on and while you are at it make sure the blocks are not to big too. Maybe you wrote the code on a new GPU 
    but your client is using an old GPU. Check both and if either is out of bound report it to your client then kindly
    exit the program.
 3. Always checking to see if you have threads working past your vector is a real pain and adds a bunch of time consumming
    if statments to your GPU code. To get around this findout how much you would have to add to your vector to make it 
    perfectly fit in your block and grid layout and pad it with zeros. Multipying zeros and adding zero do nothing to a 
    dot product. If you were luck on HW8 you kind of did this but you just got lucky because most of the time the GPU sets
    everything to zero at start up. But!!!, you don't want to put code out where you are just lucky soooo do a cudaMemset
    so you know everything is zero. Then copy up the now zero values.
 4. In HW9 we had to do the final add "reduction' on the CPU because we can't sync block. Use atomic add to get around 
    this and finish the job on the GPU. Also you will have to copy this final value down to the CPU with a cudaMemCopy.
    But!!! We are working with floats and atomics with floats can only be done on GPUs with major compute capability 3 
    or higher. Use device properties to check if this is true. And, while you are at it check to see if you have more
    than 1 GPU and if you do select the best GPU based on compute capablity.
 5. Add any additional bells and whistles to the code that you thing would make the code better and more foolproof.
*/

// Include files
#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Defines
#define N 100000 // Length of the vector
#define BLOCK_SIZE 256 // Threads in a block
float Tolerance = 0.01;

// Global variables
float *A_CPU = nullptr, *B_CPU = nullptr;
float *A_GPU = nullptr, *B_GPU = nullptr;
float *d_dot = nullptr;
float DotCPU = 0.0f, DotGPU = 0.0f;                    

dim3 BlockSize(BLOCK_SIZE, 1, 1);
dim3 GridSize(1, 1, 1);
size_t Npad = 0;
int deviceChosen = -1;

// Function prototypes
void cudaErrorCheck(const char *, int);
static inline bool isPowerOfTwo(unsigned);
static void allocateHostAndInit();
static void allocateDeviceAndCopy();
void dotProductCPU(float*, float*, int, float*);
__global__ void dotProductGPU_atomic(const float* __restrict__ a,
                                     const float* __restrict__ b,
                                     float* __restrict__ d_dot_out,
                                     int npad);

bool  check(float, float, float);
static void chooseBestDeviceOrDie();
long elaspedTime(struct timeval, struct timeval);
static void CleanUp();

// This check to see if an error happened in your CUDA code. It tell you what it thinks went wrong,
// and what file and line it occured on.
void cudaErrorCheck(const char *file, int line)
{
	cudaError_t  error;
	error = cudaGetLastError();

	if(error != cudaSuccess)
	{
		printf("\n CUDA ERROR: message = %s, File = %s, Line = %d\n", cudaGetErrorString(error), file, line);
		exit(0);
	}
}

//function to check if power of two
static inline bool isPowerOfTwo(unsigned x) {
    return x != 0 && ( (x & (x - 1)) == 0 );
}

// Allocating the memory we will be using.
static void allocateHostAndInit() {
    A_CPU = (float*)malloc(sizeof(float)*N);
    B_CPU = (float*)malloc(sizeof(float)*N);
    if (!A_CPU || !B_CPU) { printf("Host malloc failed.\n"); exit(1); }
    for (int i=0;i<N;++i) { A_CPU[i]=(float)i; B_CPU[i]=(float)(3*i); }
}

static void allocateDeviceAndCopy() {
    cudaMalloc(&A_GPU, sizeof(float)*Npad); 
	cudaErrorCheck(__FILE__, __LINE__);
    cudaMalloc(&B_GPU, sizeof(float)*Npad); 
	cudaErrorCheck(__FILE__, __LINE__);
    cudaMalloc(&d_dot, sizeof(float));   
	cudaErrorCheck(__FILE__, __LINE__);   

    cudaMemset(A_GPU, 0, sizeof(float)*Npad); 
	cudaErrorCheck(__FILE__, __LINE__);
    cudaMemset(B_GPU, 0, sizeof(float)*Npad); 
	cudaErrorCheck(__FILE__, __LINE__);
    cudaMemset(d_dot, 0, sizeof(float));     
	cudaErrorCheck(__FILE__, __LINE__); 

    cudaMemcpy(A_GPU, A_CPU, sizeof(float)*N, cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
    cudaMemcpy(B_GPU, B_CPU, sizeof(float)*N, cudaMemcpyHostToDevice); 
	cudaErrorCheck(__FILE__, __LINE__);
}

// Adding vectors a and b on the CPU then stores result in vector c.
void dotProductCPU(float *a, float *b, int n, float *out)
{
    double acc = 0.0;
    for (int i = 0; i < n; ++i) acc += (double)a[i] * (double)b[i];
    *out = (float)acc;
}

// This is the kernel. It is the function that will run on the GPU.
// It adds vectors a and b on the GPU then stores result in vector c.
__global__ void dotProductGPU_atomic(const float* __restrict__ a,const float* __restrict__ b,float* __restrict__ d_dot_out,int npad)
{	
	//shared mem
	 __shared__ float sh[BLOCK_SIZE];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float v = a[idx] * b[idx];
    sh[threadIdx.x] = v;
    __syncthreads();

    // Power-of-two reduction
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sh[threadIdx.x] += sh[threadIdx.x + stride];
        }
        __syncthreads();
    }

	//add
    if (threadIdx.x == 0) {
        atomicAdd(d_dot_out, sh[0]);
    }
}

// Checking to see if anything went wrong in the vector addition.
bool check(float cpuAnswer, float gpuAnswer, float tolerence)
{
	double percentError;
	
	percentError = abs((gpuAnswer - cpuAnswer)/(cpuAnswer))*100.0;
	printf("\n\n percent error = %lf\n", percentError);
	
	if(percentError < Tolerance) 
	{
		return(true);
	}
	else 
	{
		return(false);
	}
}

static void chooseBestDeviceOrDie() {
    int count = 0;
    cudaGetDeviceCount(&count);
    cudaErrorCheck(__FILE__, __LINE__);
    if (count == 0) {
        printf("No CUDA-capable GPU detected.\n");
        exit(1);
    }

    int best = 0;
    int bestMajor = -1, bestMinor = -1, bestSMs = -1;

    for (int i = 0; i < count; ++i) {
        cudaDeviceProp p;
        cudaGetDeviceProperties(&p, i);
        // Prefer highest compute capability, then SMs
        if (p.major > bestMajor ||
            (p.major == bestMajor && p.minor > bestMinor) ||
            (p.major == bestMajor && p.minor == bestMinor && p.multiProcessorCount > bestSMs)) {
            best = i;
            bestMajor = p.major;
            bestMinor = p.minor;
            bestSMs   = p.multiProcessorCount;
        }
    }

    cudaSetDevice(best);
    cudaErrorCheck(__FILE__, __LINE__);
    deviceChosen = best;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, deviceChosen);

    printf("Using GPU %d:\n",deviceChosen);

    //check compute capability
    if (prop.major < 3) {
        printf("This GPU has compute capability %d.%d; need >= 3.0 for float atomicAdd per assignment.\n",
               prop.major, prop.minor);
        exit(1);
    }

    //check block size is a power of two
    if (!isPowerOfTwo(BLOCK_SIZE)) {
        printf("BLOCK_SIZE=%d is not a power of two", BLOCK_SIZE);
        exit(1);
    }

    //choose GridSize to cover N
    // Start with the minimal grid that covers N
    unsigned blocksNeeded = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    //check device limits
    if (BLOCK_SIZE > prop.maxThreadsPerBlock) {
        printf("BLOCK_SIZE=%d exceeds device maxThreadsPerBlock=%d on %s.\n",
               BLOCK_SIZE, prop.maxThreadsPerBlock, prop.name);
        exit(1);
    }

    //check grid dimension vs maxGridSize
    if ((int)blocksNeeded > prop.maxGridSize[0]) {
        printf("GridSize.x=%u exceeds device maxGridSize.x=%d on %s.\n",
               blocksNeeded, prop.maxGridSize[0], prop.name);
        exit(1);
    }

    GridSize.x = blocksNeeded;
    GridSize.y = 1;
    GridSize.z = 1;

    //compute padded length exactly equal to GridSize.x * BLOCK_SIZE
    Npad = (size_t)GridSize.x * (size_t)BLOCK_SIZE;

    printf("Logical N=%d, BLOCK_SIZE=%d, GridSize.x=%u -> Npad=%zu\n",
           N, BLOCK_SIZE, GridSize.x, Npad);
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
static void CleanUp() {
    if (A_CPU) free(A_CPU);
    if (B_CPU) free(B_CPU);
    if (A_GPU) cudaFree(A_GPU);
    if (B_GPU) cudaFree(B_GPU);
    if (d_dot) cudaFree(d_dot);
    cudaErrorCheck(__FILE__, __LINE__);
}

int main()
{
	timeval start, end;
	long timeCPU, timeGPU;
	//float localC_CPU, localC_GPU;
	
	//pick best device, set Grid/Block, compute Npad
    chooseBestDeviceOrDie();
	
	// Allocating the memory you will need.
	allocateHostAndInit();
	
	// Adding on the CPU
	gettimeofday(&start, NULL);
	dotProductCPU(A_CPU, B_CPU, N, &DotCPU);
	gettimeofday(&end, NULL);
	timeCPU = elaspedTime(start, end);

	//allocating mem to gpu
	allocateDeviceAndCopy();
	
	// Adding on the GPU
	gettimeofday(&start, NULL);
	
	dotProductGPU_atomic<<<GridSize, BlockSize>>>(A_GPU, B_GPU, d_dot, (int)Npad);
	cudaErrorCheck(__FILE__, __LINE__);
	
	// Copy Memory from GPU to CPU	
	cudaMemcpy(&DotGPU, d_dot, sizeof(float), cudaMemcpyDeviceToHost);
	cudaErrorCheck(__FILE__, __LINE__);
	
	// Making sure the GPU and CPU wiat until each other are at the same place.
	cudaDeviceSynchronize();
	cudaErrorCheck(__FILE__, __LINE__);

	gettimeofday(&end, NULL);
	timeGPU = elaspedTime(start, end);
	
	// Checking to see if all went correctly.
	if(check(DotCPU, DotGPU, Tolerance) == false)
	{
		printf("\n\n Something went wrong in the GPU dot product.\n");
	}
	else
	{
		printf("\n\n You did a dot product correctly on the GPU");
		printf("\n The time it took on the CPU was %ld microseconds", timeCPU);
		printf("\n The time it took on the GPU was %ld microseconds", timeGPU);
	}
	
	// Your done so cleanup your room.	
	CleanUp();	
	
	// Making sure it flushes out anything in the print buffer.
	printf("\n\n");
	
	return(0);
}


