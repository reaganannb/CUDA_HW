// Name:Reagan Burleson
// Device query
// nvcc HW5.cu -o temp
/*
 What to do:
 This code prints out useful information about the GPU(s) in your machine, 
 but there is much more data available in the cudaDeviceProp structure.

 Extend this code so that it prints out all the information about the GPU(s) in your system. 
 Also, and this is the fun part, be prepared to explain what each piece of information means. 
*/

// Include files
#include <stdio.h>

// Defines

// Global variables

// Function prototypes
void cudaErrorCheck(const char*, int);

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

int main()
{
	cudaDeviceProp prop;

	int count;
	cudaGetDeviceCount(&count);
	cudaErrorCheck(__FILE__, __LINE__);
	printf(" You have %d GPUs in this machine\n", count);
	
	for (int i=0; i < count; i++) {
		cudaGetDeviceProperties(&prop, i);
		cudaErrorCheck(__FILE__, __LINE__);
		printf(" ---General Information for device %d ---\n", i);
		printf("Name: %s\n", prop.name);
		printf("Compute capability: %d.%d\n", prop.major, prop.minor);
		printf("Clock rate: %d\n", prop.clockRate);
		printf("Device copy overlap: ");
		if (prop.deviceOverlap) printf("Enabled\n");
		else printf("Disabled\n");
		printf("Kernel execution timeout : ");
		if (prop.kernelExecTimeoutEnabled) printf("Enabled\n");
		else printf("Disabled\n");
		printf("Compute Mode: %d\n", prop.computeMode);
		switch (prop.computeMode) {
    		case cudaComputeModeDefault:
				printf("   Multiple contexts are allowed (Default)\n");
				break;
			case cudaComputeModeExclusive:
				printf("   Only one context can be active on this device at a time\n");
				break;
			case cudaComputeModeProhibited:
				printf("   No contexts can be created on this device\n");
				break;
			case cudaComputeModeExclusiveProcess:
				printf("   Only one context per process can be active on this device\n");
				break;
			default:
				printf("   Unknown compute mode\n");
				break;
		}
		printf("Concurrent Kernels : ");
		if (prop.concurrentKernels) printf("Enabled\n");
		else printf("Disabled\n");
		printf("Total global mem: %ld\n", prop.totalGlobalMem);
		printf("ECC Enabled : ");
		if (prop.ECCEnabled) printf("Enabled\n");
		else printf("Disabled\n");
		printf("Total global mem: %ld\n", prop.totalGlobalMem);
		printf("Integrated : ");
		if (prop.integrated) printf("Enabled\n");
		else printf("Disabled\n");
		printf("Total global mem: %ld\n", prop.totalGlobalMem);

		printf(" ---Memory Information for device %d ---\n", i);
		printf("Map Host Memory : ");
		if (prop.canMapHostMemory) printf("Enabled\n");
		else printf("Disabled\n");
		printf("PCI bus ID: %d\n", prop.pciBusID);
		printf("PCI device ID: %d\n", prop.pciDeviceID);
		printf("Surface Alignment: %zu bytes\n",prop.surfaceAlignment);
		printf("TCC Driver: %d\n",prop.tccDriver);
		printf("Total global mem: %ld\n", prop.totalGlobalMem);
		printf("Total constant Mem: %ld\n", prop.totalConstMem);
		printf("Max mem pitch: %ld\n", prop.memPitch);
		printf("Texture Alignment: %ld\n", prop.textureAlignment);
		printf(" ---MP Information for device %d ---\n", i);
		printf("Multiprocessor count : %d\n", prop.multiProcessorCount); //Number of SMs on the GPU
		printf("Shared mem per mp: %ld\n", prop.sharedMemPerBlock);
		printf("Registers per mp: %d\n", prop.regsPerBlock);
		printf("Threads in warp: %d\n", prop.warpSize);
		printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
		printf("Max thread dimensions: (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
		printf("Max grid dimensions: (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
		printf("Max texture in 1D: %d\n", prop.maxTexture1D);
		printf("Max texture in 2D: %d\n", prop.maxTexture2D);
		printf("Max texture in 3D: %d\n", prop.maxTexture3D);

		printf("\n");
	}	
	return(0);
}

