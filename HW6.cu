// Name:Reagan Burleson
// Simple Julia CPU.
// nvcc HW6.cu -o temp -lglut -lGL
// glut and GL are openGL libraries.
/*
 What to do:
 This code displays a simple Julia fractal using the CPU.
 Rewrite the code so that it uses the GPU to create the fractal. 
 Keep the window at 1024 by 1024.
*/

// Include files
#include <stdio.h>
#include <GL/glut.h>

#include <math.h>
#include <cuda_runtime.h>

// Defines
#define MAXMAG 10.0 // If you grow larger than this, we assume that you have escaped.
#define MAXITERATIONS 200 // If you have not escaped after this many attempts, we assume you are not going to escape.
#define A  -0.824	//Real part of C
#define B  -0.1711	//Imaginary part of C

// Global variables
unsigned int WindowWidth = 1024;
unsigned int WindowHeight = 1024;

float XMin = -2.0f;
float XMax =  2.0f;
float YMin = -2.0f;
float YMax =  2.0f;

// Function prototypes
void cudaErrorCheck(const char*, int);
__host__ __device__ float escapeOrNotColor(float, float);
__global__ void juliaKernel(float*, int, int, float, float, float, float);
void display(void);

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

__host__ __device__ float escapeOrNotColor (float x, float y) 
{
	float mag,tempX;
	int count;
	
	int maxCount = MAXITERATIONS;
	float maxMag = MAXMAG;
	
	count = 0;
	mag = sqrt(x*x + y*y);;
	while (mag < maxMag && count < maxCount) 
	{	
		tempX = x; //We will be changing the x but we need its old value to find y.
		x = x*x - y*y + A;
		y = (2.0f * tempX * y) + B;
		mag = sqrt(x*x + y*y);
		count++;
	}
	if(count < maxCount) 
	{
		return(0.0f);
	}
	else
	{
		return(1.0f);
	}
}

__global__ void juliaKernel(float* outRGB,
                            int width, int height,
                            float xmin, float ymin,
                            float stepX, float stepY)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= width || iy >= height) return;

    float x = xmin + ix * stepX;
    float y = ymin + iy * stepY;

    float r = escapeOrNotColor(x, y);

    int k = (iy * width + ix) * 3;
    outRGB[k + 0] = r;     // Red
    outRGB[k + 1] = 0.0f;  // Green
    outRGB[k + 2] = 0.0f;  // Blue
}

void display(void) 
{ 
    size_t n = (size_t)WindowWidth * WindowHeight * 3;
    float *pixels = (float*)malloc(n * sizeof(float));
    float *d_pixels = nullptr;

    float stepSizeX = (XMax - XMin) / (float)WindowWidth;
    float stepSizeY = (YMax - YMin) / (float)WindowHeight;

    cudaMalloc(&d_pixels, n * sizeof(float));
    cudaErrorCheck(__FILE__, __LINE__);

    dim3 block(16, 16);
    dim3 grid((WindowWidth  + block.x - 1) / block.x,
              (WindowHeight + block.y - 1) / block.y);

    juliaKernel<<<grid, block>>>(d_pixels,
                                 (int)WindowWidth, (int)WindowHeight,
                                 XMin, YMin, stepSizeX, stepSizeY);
	cudaDeviceSynchronize();
    cudaErrorCheck(__FILE__, __LINE__);

    cudaMemcpy(pixels, d_pixels, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaErrorCheck(__FILE__, __LINE__);

    glDrawPixels(WindowWidth, WindowHeight, GL_RGB, GL_FLOAT, pixels); 
    glFlush(); 

    cudaFree(d_pixels);
    free(pixels);
}

int main(int argc, char** argv)
{ 
   	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);
   	glutInitWindowSize(WindowWidth, WindowHeight);
	glutCreateWindow("Fractals--Man--Fractals");
   	glutDisplayFunc(display);
   	glutMainLoop();
}


