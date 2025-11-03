// Name: Reagan Burleson
// GPU random walk. 
// nvcc 16GPURandomWalk.cu -o temp

/*
 What to do:
 This code runs a random walk for 10,000 steps on the CPU.

 1. Use cuRAND to run 20 random walks simultaneously on the GPU, each with a different seed.
    Print out all 20 final positions.

 2. Use cudaMallocManaged(&variable, amount_of_memory_needed);
    This allocates unified memory, which is automatically managed between the CPU and GPU.
    You lose some control over placement, but it saves you from having to manually copy data
    to and from the GPU.
*/

/*
 Purpose:
 To learn how to use cuRAND and unified memory.
*/

/*
 Note:
 The maximum signed int value is 2,147,483,647, so the maximum unsigned int value is 4,294,967,295.

 RAND_MAX is guaranteed to be at least 32,767. When I checked it on my laptop (10/6/2025), it was 2,147,483,647.
 rand() returns a value in [0, RAND_MAX]. It actually generates a list of pseudo-random numbers that depends on the seed.
 This list eventually repeats (this is called its period). The period is usually 2³¹ = 2,147,483,648,
 but it may vary by implementation.

 Because RAND_MAX is odd on this machine and 0 is included, there is no exact middle integer.
 Casting to float as in (float)RAND_MAX / 2.0 divides the range evenly.
 Using integer division (RAND_MAX / 2) would bias results slightly toward the positive side by one value out of 2,147,483,647.

 I know this is splitting hares (sorry, rabbits), but I'm just trying to be as accurate as possible.
 You might do this faster with a clever integer approach, but I’m using floats here for clarity.
*/

// Include files
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

static const int kNumWalks = 20;
static const int NumberOfRandomSteps = 10000;
float MidPoint = (float)RAND_MAX/2.0f;

// Function prototypes
void CUDA_CHECK();
int getRandomDirection();
int main(int, char**);

#define CUDA_CHECK(call)                                                                      \
    do {                                                                                      \
        cudaError_t _err = (call);                                                            \
        if (_err != cudaSuccess) {                                                            \
            fprintf(stderr, "CUDA error %s at %s:%d -> %s\n", #call, __FILE__, __LINE__,     \
                    cudaGetErrorString(_err));                                                \
            std::exit(EXIT_FAILURE);                                                          \
        }                                                                                     \
    } while (0)

__global__ void randomWalkKernel(int2* finalPos, int steps, unsigned long long baseSeed)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= kNumWalks) return;

    //declare cuRand object for each thread
    curandStatePhilox4_32_10_t rng;
    //init cuRand (seed, sequence, offset=0)
    curand_init(baseSeed + tid, 0ULL, 0ULL, &rng);

    int x = 0, y = 0;

    //takes random float between 0,1 if >0.5 step right else step left 
    for (int i = 0; i < steps; ++i) {
		//calling curand for rand number
        float r1 = curand_uniform(&rng);
        float r2 = curand_uniform(&rng);
        x += (r1 < 0.5f) ? -1 : 1;
        y += (r2 < 0.5f) ? -1 : 1;
    }

    finalPos[tid] = make_int2(x, y);
}

int getRandomDirection()
{	
	int randomNumber = rand();
	
	if(randomNumber < MidPoint) return(-1);
	else return(1);
}

int main(int argc, char** argv)
{
	printf(" RAND_MAX for this implementation is = %d \n", RAND_MAX);

	//Allocate Unified Memory for the 20 final positions
    int2* finalPositions = nullptr;
    CUDA_CHECK(cudaMallocManaged(&finalPositions, kNumWalks * sizeof(int2)));

	//Blocks config
	dim3 block(32);
    dim3 grid((kNumWalks + block.x - 1) / block.x);

	//seed
	unsigned long long baseSeed = static_cast<unsigned long long>(time(nullptr));

	randomWalkKernel<<<grid, block>>>(finalPositions, NumberOfRandomSteps, baseSeed);
    CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());
	
	for (int i = 0; i < kNumWalks; ++i) {
        printf("Walk %2d final position = (%d,%d)\n", i, finalPositions[i].x, finalPositions[i].y);
    }
	
	// Cleanup
    CUDA_CHECK(cudaFree(finalPositions));
	
	return 0;
}

