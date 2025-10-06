// Name: Reagan Burleson
// Ray tracing
// nvcc 12RayTracerWithConstantMemory.cu -o temp -lglut -lGL -lm
/*
 What to do:
 This code creates a random set of spheres and uses ray tracing to create a picture of them to be 
 displayed on the screen.
 1: The spheres you created on the CPU do not change so send them up to the GPU and put them into constant memory.
 2: Use CUDA events to time your code.
*/

// Include files
#include <GL/glut.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

#include <cuda_runtime.h>

// Defines
#define WINDOWWIDTH 1024
#define WINDOWHEIGHT 1024
#define XMIN -1.0f
#define XMAX 1.0f
#define YMIN -1.0f
#define YMAX 1.0f
#define ZMIN -1.0f
#define ZMAX 1.0f
#define NUMSPHERES 100
#define MAXRADIUS 0.2  // The window is a 2 by 2 square.

// Local structures
struct sphereStruct 
{
	float r,b,g; // Sphere color
	float radius;
	float x,y,z; // Sphere center
};

// Globals variables
static int Window;
unsigned int WindowWidth = WINDOWWIDTH;
unsigned int WindowHeight = WINDOWHEIGHT;
dim3 BlockSize, GridSize;
float *PixelsCPU, *PixelsGPU; 
sphereStruct *SpheresCPU, *SpheresGPU;

// Timing (CUDA events)
cudaEvent_t evtKernelStart, evtKernelStop, evtCopyStart, evtCopyStop;

// Function prototypes
void cudaErrorCheck(const char *, int);
void Display();
void idle();
void KeyPressed(unsigned char , int , int );
__device__ float hit(float , float , float *, float , float , float , float );
__global__ void makeSphersBitMap(float *);
void makeRandomSpheres();
void makeBitMap();
void paintScreen();
void setup();

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

void display()
{
	makeBitMap();	
}

void KeyPressed(unsigned char key, int x, int y)
{	
	if(key == 'q')
	{
		glutDestroyWindow(Window);
		printf("\nw Good Bye\n");
		exit(0);
	}
}

__device__ float hit(float pixelx, float pixely, float *dimingValue, sphereStruct sphere)
{
	float dx = pixelx - sphere.x;  //Distance from ray to sphere center in x direction
	float dy = pixely - sphere.y;  //Distance from ray to sphere center in y direction
	float r2 = sphere.radius*sphere.radius;
	if(dx*dx + dy*dy < r2) // if the ray hits the sphere, then we need to find distance
	{
		float dz = sqrtf(r2 - dx*dx - dy*dy); // Distance from ray to edge of sphere?
		*dimingValue = dz/sphere.radius; // n is value between 0 and 1 used for darkening points near edge.
		return dz + sphere.z; //  Return the distance to be scaled by
	}
	return (ZMIN- 1.0); //If the ray doesn't hit anything return a number behind the box.
}

__global__ void makeSphersBitMap(float *pixels)
{
	float stepSizeX = (XMAX - XMIN)/((float)WINDOWWIDTH - 1);
	float stepSizeY = (YMAX - YMIN)/((float)WINDOWHEIGHT - 1);
	
	// Asigning each thread a pixel
	float pixelx = XMIN + threadIdx.x*stepSizeX;
	float pixely = YMIN + blockIdx.x*stepSizeY;
	
	// Finding this pixels location in memory
	int id = 3*(threadIdx.x + blockIdx.x*blockDim.x);
	
	//initialize rgb values for each pixel to zero (black)
	float pixelr = 0.0f;
	float pixelg = 0.0f;
	float pixelb = 0.0f;
	float hitValue;
	float dimingValue;
	float maxHit = ZMIN -1.0f; // Initializing it to be out of the back of the box.
	#pragma unroll
    for (int i = 0; i < NUMSPHERES; i++)
    {
        hitValue = hit(pixelx, pixely, &dimingValue, cSpheres[i]);
        if (maxHit < hitValue)
        {
            pixelr = cSpheres[i].r * dimingValue;
            pixelg = cSpheres[i].g * dimingValue;
            pixelb = cSpheres[i].b * dimingValue;
            maxHit = hitValue;
        }
    }
	
	pixels[id] = pixelr;
	pixels[id+1] = pixelg;
	pixels[id+2] = pixelb;
}

void makeRandomSpheres()
{	
	float rangeX = XMAX - XMIN;
	float rangeY = YMAX - YMIN;
	float rangeZ = ZMAX - ZMIN;
	
	for(int i = 0; i < NUMSPHERES; i++)
	{
		SpheresCPU[i].x = (rangeX*(float)rand()/RAND_MAX) + XMIN;
		SpheresCPU[i].y = (rangeY*(float)rand()/RAND_MAX) + YMIN;
		SpheresCPU[i].z = (rangeZ*(float)rand()/RAND_MAX) + ZMIN;
		SpheresCPU[i].r = (float)rand()/RAND_MAX;
		SpheresCPU[i].g = (float)rand()/RAND_MAX;
		SpheresCPU[i].b = (float)rand()/RAND_MAX;
		SpheresCPU[i].radius = MAXRADIUS*(float)rand()/RAND_MAX;
	}
	//copying to constant mem
	cudaMemcpyToSymbol(cSpheres, SpheresCPU, NUMSPHERES * sizeof(sphereStruct));
    cudaErrorCheck(__FILE__, __LINE__);
}	

void makeBitMap()
{	
	//kernel timing
    cudaEventRecord(evtKernelStart, 0);
    makeSpheresBitMap<<<GridSize, BlockSize>>>(PixelsGPU);
    cudaErrorCheck(__FILE__, __LINE__);
    cudaEventRecord(evtKernelStop, 0);
    cudaEventSynchronize(evtKernelStop);

	float kernelMs = 0.0f;
    cudaEventElapsedTime(&kernelMs, evtKernelStart, evtKernelStop);

	//host copy
	cudaEventRecord(evtCopyStart, 0);
    cudaMemcpyAsync(PixelsCPU, PixelsGPU, WINDOWWIDTH * WINDOWHEIGHT * 3 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaErrorCheck(__FILE__, __LINE__);
    cudaEventRecord(evtCopyStop, 0);
    cudaEventSynchronize(evtCopyStop);

	float copyMs = 0.0f;
    cudaEventElapsedTime(&copyMs, evtCopyStart, evtCopyStop);

	//print every frame
    static int frame = 0;
    if ((frame++ % 60) == 0) {
        printf("Kernel: %.3f ms, D->H copy: %.3f ms\n", kernelMs, copyMs);
        fflush(stdout);
    }
	
	paintScreen();
}

void paintScreen()
{
	//Putting pixels on the screen.
	glDrawPixels(WINDOWWIDTH, WINDOWHEIGHT, GL_RGB, GL_FLOAT, PixelsCPU); 
	glFlush();
}

void setup()
{
	//We need the 3 because each pixel has a red, green, and blue value.
	PixelsCPU = (float *)malloc(WINDOWWIDTH*WINDOWHEIGHT*3*sizeof(float));
	cudaMalloc(&PixelsGPU,WINDOWWIDTH*WINDOWHEIGHT*3*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	
	SpheresCPU= (sphereStruct*)malloc(NUMSPHERES*sizeof(sphereStruct));
	
	//Threads in a block
	if(WINDOWWIDTH > 1024)
	{
	 	printf("The window width is too large to run with this program\n");
	 	printf("The window width must be less than 1024.\n");
	 	printf("Good Bye and have a nice day!\n");
	 	exit(0);
	}
	BlockSize.x = WINDOWWIDTH;
	BlockSize.y = 1;
	BlockSize.z = 1;
	
	//Blocks in a grid
	GridSize.x = WINDOWHEIGHT;
	GridSize.y = 1;
	GridSize.z = 1;
	
	// Seading the random number generater.
	time_t t;
	srand((unsigned) time(&t));

	//create events
    cudaEventCreate(&evtKernelStart);
    cudaEventCreate(&evtKernelStop);
    cudaEventCreate(&evtCopyStart);
    cudaEventCreate(&evtCopyStop);
}

int main(int argc, char** argv)
{ 
	setup();
	makeRandomSpheres();
   	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);
   	glutInitWindowSize(WINDOWWIDTH, WINDOWHEIGHT);
	Window = glutCreateWindow("Random Spheres");
	glutKeyboardFunc(KeyPressed);
   	glutDisplayFunc(display);
   	glutMainLoop();
}

