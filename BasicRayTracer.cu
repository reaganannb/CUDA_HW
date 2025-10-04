// Name: Reagan Burleson
// Ray tracing
// nvcc HW12.cu -o temp -lglut -lGL -lm
/*
 What to do:
 This code creates a random set of spheres and uses ray tracing to create a picture of them to be 
 displayed on the screen. Go through the code and understand it so you can explain it to the class.
*/

// Include files
#include <GL/glut.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

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

// Function prototypes
void cudaErrorCheck(const char *, int);
void Display();
void idle();
void KeyPressed(unsigned char , int , int );
__device__ float hit(float , float , float *, float , float , float , float );
__global__ void makeSphersBitMap(float *, sphereStruct *);
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

__global__ void makeSphersBitMap(float *pixels, sphereStruct *sphereInfo)
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
	for(int i = 0; i < NUMSPHERES; i++)
	{
		hitValue = hit(pixelx, pixely, &dimingValue, sphereInfo[i]);
		// do we hit any spheres? If so, how close are we to the center? (i.e. n)
		if(maxHit < hitValue)
		{
			// Setting the RGB value of the sphere but also diming it as it gets close to the side of the sphere.
			pixelr = sphereInfo[i].r * dimingValue; 	
			pixelg = sphereInfo[i].g * dimingValue;	
			pixelb = sphereInfo[i].b * dimingValue; 	
			maxHit = hitValue; // reset maxHit value to be the current closest sphere
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
}	

void makeBitMap()
{	
	cudaMemcpy(SpheresGPU, SpheresCPU, NUMSPHERES*sizeof(sphereStruct), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	
	makeSphersBitMap<<<GridSize, BlockSize>>>(PixelsGPU, SpheresGPU);
	cudaErrorCheck(__FILE__, __LINE__);
	
	cudaMemcpyAsync(PixelsCPU, PixelsGPU, WINDOWWIDTH*WINDOWHEIGHT*3*sizeof(float), cudaMemcpyDeviceToHost);
	cudaErrorCheck(__FILE__, __LINE__);
	
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
	cudaMalloc(&SpheresGPU, NUMSPHERES*sizeof(sphereStruct));
	cudaErrorCheck(__FILE__, __LINE__);
	
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

