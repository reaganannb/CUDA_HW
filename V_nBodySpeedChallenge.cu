// Name: Reagan Burleson
// Optimizing nBody GPU code. 
// nvcc -O3 --use_fast_math -arch=sm_75 -Xptxas -O3,-dlcm=ca,-v  V_nBodySpeedChallenge.cu -o temp -lglut -lm -lGLU -lGL


/*
 What to do:
 This is some lean n-body code that runs on the GPU for any number of bodies (within reason). Take this code and make it 
 run as fast as possible using any tricks you know or can find (Like using NVIDIA Nsight Systems). Keep the same general 
 format so we can time it and compare it with others' code. This will be a competition.
 
 First place: 20 extra points on this HW
 Second place: 15 extra points on this HW
 Third place: 10 extra points on this HW
 
 To focus more on new ideas rather than just using a bunch of if statements to avoid going out of bounds, N will be a power 
 of 2 and 256 < N < 262,144. Put a check in your code to make sure this is true. The code most run on any power of 2 bodies
 also the final picture most look close to the same as it did before the speedup or something went wrong in the code.

 Note: The code takes two arguments as inputs:
 1. The number of bodies to simulate.
 2. Whether to draw sub-arrangements of the bodies during the simulation (1), or only the first and last arrangements (0).
 
 Use this code (before your changes) as the baseline code to check your nbody speedup.
*/

/*
 Purpose:
 To use what you have learned in this course to optimize code with the add of NVIDIA Nsight Systems.
*/

// Include files
#include <GL/glut.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

// Defines
#define BLOCK_SIZE 256
#define PI 3.14159265359
#define DRAW_RATE 10

// This is to create a Lennard-Jones type function G/(r^p) - H(r^q). (p < q) p has to be less than q.
// In this code we will keep it a p = 2 and q = 4 problem. The diameter of a body is found using the general
// case so it will be more robust but in the code leaving it as a set 2, 4 problem make the coding much easier.
#define G 10.0f
#define H 10.0f
#define LJP  2.0
#define LJQ  4.0

#define DT 0.0001
#define RUN_TIME 1.0

#define STEPS_PER_LAUNCH 4

// Globals
int N, DrawFlag;
float3 *P, *V;
float *M; 
float3 *PGPU, *VGPU;
float *MGPU;
float GlobeRadius, Diameter, Radius;
float Damp;
dim3 BlockSize;
dim3 GridSize;
size_t ShmemBytes = 0;


// Function prototypes
void cudaErrorCheck(const char *, int);
void keyPressed(unsigned char, int, int);
long elaspedTime(struct timeval, struct timeval);
void drawPicture();
void timer();
void setup();
__global__ void stepBodies(float3 *__restrict__,
                           float3 *__restrict__,
                           const float *__restrict__,
                           const float *__restrict__,
                           float, float, float, float, float, int);
void nBody();
int main(int, char**);

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

void keyPressed(unsigned char key, int x, int y)
{
	if(key == 's')
	{
		printf("\n The simulation is running.\n");
		timer();
	}
	
	if(key == 'q')
	{
		exit(0);
	}
}

static inline int isPowerOfTwo(int x){ return x > 0 && (x & (x-1)) == 0; }

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

void drawPicture()
{
	glClear(GL_COLOR_BUFFER_BIT);
	glClear(GL_DEPTH_BUFFER_BIT);
	
	cudaMemcpyAsync(P, PGPU, N*sizeof(float3), cudaMemcpyDeviceToHost);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaStreamSynchronize(0);
	
	glColor3d(1.0,1.0,0.5);
	for(int i=0; i<N; i++)
	{
		glPushMatrix();
		glTranslatef(P[i].x, P[i].y, P[i].z);
		glutSolidSphere(Radius,20,20);
		glPopMatrix();
	}
	
	glutSwapBuffers();
}

void timer()
{	
	timeval start, end;
	long computeTime;
	
	drawPicture();
	gettimeofday(&start, NULL);
    
	nBody();
    cudaDeviceSynchronize();
	cudaErrorCheck(__FILE__, __LINE__);
    
	gettimeofday(&end, NULL);
    drawPicture();
    	
	computeTime = elaspedTime(start, end);
	printf("\n The compute time was %ld microseconds.\n\n", computeTime);
}

void setup()
{
	float randomAngle1, randomAngle2, randomRadius;
	float d, dx, dy, dz;
	int test;
    	
    BlockSize   = dim3(BLOCK_SIZE, 1, 1);
	GridSize    = dim3(N / BLOCK_SIZE, 1, 1);  
	ShmemBytes = BLOCK_SIZE * (sizeof(float3) + sizeof(float));

	
	Damp = 0.5;
	
	cudaHostAlloc((void**)&P, N*sizeof(float3), cudaHostAllocDefault);
    V = (float3*)malloc(N*sizeof(float3));
    M = (float*)  malloc(N*sizeof(float));
	
	cudaMalloc(&MGPU,N*sizeof(float));
    cudaErrorCheck(__FILE__, __LINE__);
    cudaMalloc(&PGPU,N*sizeof(float3));
    cudaErrorCheck(__FILE__, __LINE__);
    cudaMalloc(&VGPU,N*sizeof(float3));
    cudaErrorCheck(__FILE__, __LINE__);
    	
	Diameter = pow(H/G, 1.0/(LJQ - LJP)); // This is the value where the force is zero for the L-J type force.
	Radius = Diameter/2.0;
	
	// Using the radius of a body and a 68% packing ratio to find the radius of a global sphere that should hold all the bodies.
	// Then we double this radius just so we can get all the bodies setup with no problems. 
	float totalVolume = float(N)*(4.0/3.0)*PI*Radius*Radius*Radius;
	totalVolume /= 0.68;
	float totalRadius = pow(3.0*totalVolume/(4.0*PI), 1.0/3.0);
	GlobeRadius = 2.0*totalRadius;
	
	// Randomly setting these bodies in the glaobal sphere and setting the initial velosity, initial force, and mass.
	for(int i = 0; i < N; i++)
	{
		test = 0;
		while(test == 0)
		{
			// Get random position.
			randomAngle1 = ((float)rand()/(float)RAND_MAX)*2.0*PI;
			randomAngle2 = ((float)rand()/(float)RAND_MAX)*PI;
			randomRadius = ((float)rand()/(float)RAND_MAX)*GlobeRadius;
			P[i].x = randomRadius*cos(randomAngle1)*sin(randomAngle2);
			P[i].y = randomRadius*sin(randomAngle1)*sin(randomAngle2);
			P[i].z = randomRadius*cos(randomAngle2);
			
			// Making sure the balls centers are at least a diameter apart.
			// If they are not throw these positions away and try again.
			test = 1;
			for(int j = 0; j < i; j++)
			{
				dx = P[i].x-P[j].x;
				dy = P[i].y-P[j].y;
				dz = P[i].z-P[j].z;
				d = sqrt(dx*dx + dy*dy + dz*dz);
				if(d < Diameter) { test = 0; break; }
			}
		}
	
		V[i].x = V[i].y = V[i].z = 0.0f;
        M[i]   = 1.0f;
	}
	
	cudaMemcpyAsync(PGPU, P, N*sizeof(float3), cudaMemcpyHostToDevice);
    cudaErrorCheck(__FILE__, __LINE__);
    cudaMemcpyAsync(VGPU, V, N*sizeof(float3), cudaMemcpyHostToDevice);
    cudaErrorCheck(__FILE__, __LINE__);
    cudaMemcpyAsync(MGPU, M, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaErrorCheck(__FILE__, __LINE__);
	
	printf("\n To start timing go to the nBody window and type s.\n");
	printf("\n To quit type q in the nBody window.\n");
}

__global__ __launch_bounds__(BLOCK_SIZE, 2)
void stepBodies(float3 *__restrict__ p,
                float3 *__restrict__ v,
                const float *__restrict__ m,
                float g, float h,
                float damp, float dt,
                float t, int n,int useHalfKick)
{
    extern __shared__ unsigned char smem[];
    float3* tileP = (float3*)smem;
    float*  tileM = (float*)(tileP + blockDim.x);

    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= n) return;   

    float3 pi = p[i];
    float3 vi = v[i];
    float   mi = m[i];

    float3 fi; fi.x = 0.0f; fi.y = 0.0f; fi.z = 0.0f;

    for (int tile = 0; tile < n; tile += blockDim.x) {
        int j = tile + threadIdx.x;

        // no if(j<n) because N is multiple of blockDim.x
        tileP[threadIdx.x] = p[j];
        tileM[threadIdx.x] = m[j];

        __syncthreads();

        #pragma unroll 8
		for (int k = 0; k < blockDim.x; ++k) {
			int idx = tile + k;
			if (idx == i) continue;

			float dx = tileP[k].x - pi.x;
			float dy = tileP[k].y - pi.y;
			float dz = tileP[k].z - pi.z;

			float d2     = dx*dx + dy*dy + dz*dz + 1e-8f;
			float inv_d  = rsqrtf(d2);
			float inv_d2 = inv_d * inv_d;
			float inv_d4 = inv_d2 * inv_d2;

			float mk    = tileM[k];
			float mm    = mi * mk;
			float coeff = g * inv_d2 - h * inv_d4;
			float s     = (mm * coeff) * inv_d;

			fi.x += s * dx;
			fi.y += s * dy;
			fi.z += s * dz;
		}

        __syncthreads();
    }

    float inv_m = 1.0f / mi;
    float ax = (fi.x - damp * vi.x) * inv_m;
    float ay = (fi.y - damp * vi.y) * inv_m;
    float az = (fi.z - damp * vi.z) * inv_m;

    float scale = dt;
	if (useHalfKick && t == 0.0f) {
		scale = 0.5f * dt;
	}

    vi.x += ax * scale;
    vi.y += ay * scale;
    vi.z += az * scale;

    pi.x += vi.x * dt;
    pi.y += vi.y * dt;
    pi.z += vi.z * dt;

    v[i] = vi;
    p[i] = pi;
}




void nBody()
{
    int   drawCount = 0;
    float t         = 0.0f;
    float dt        = 0.0001f;
	int useHalfKick = DrawFlag ? 1 : 0;


    while (t < RUN_TIME)
    {
        stepBodies<<<GridSize, BlockSize, ShmemBytes>>>(
            PGPU, VGPU, MGPU,
            G, H,
            Damp, dt,
            t, N, useHalfKick
        );
		
        cudaErrorCheck(__FILE__, __LINE__);

        if (drawCount == DRAW_RATE) {
            if (DrawFlag) { drawPicture(); }
            drawCount = 0;
        }

        t += dt;
        drawCount++;
    }
}


int main(int argc, char** argv)
{
	if( argc < 3)
	{
		printf("\n You need to enter the number of bodies (an int)"); 
		printf("\n and if you want to draw the bodies as they move (1 draw, 0 don't draw),");
		printf("\n on the comand line.\n"); 
		exit(0);
	}
	else
	{
		N = atoi(argv[1]);
		DrawFlag = atoi(argv[2]);
	}

	if (!isPowerOfTwo(N) || N <= 256 || N >= 262144) {
        printf("\n N must be a power of 2 with 256 < N < 262,144. You gave %d.\n", N);
        exit(0);
    }

	cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
	
	setup();
	
	int XWindowSize = 1000;
	int YWindowSize = 1000;
	
	glutInit(&argc,argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_DEPTH | GLUT_RGB);
	glutInitWindowSize(XWindowSize,YWindowSize);
	glutInitWindowPosition(0,0);
	glutCreateWindow("nBody Challenge");
	GLfloat light_position[] = {1.0, 1.0, 1.0, 0.0};
	GLfloat light_ambient[]  = {0.0, 0.0, 0.0, 1.0};
	GLfloat light_diffuse[]  = {1.0, 1.0, 1.0, 1.0};
	GLfloat light_specular[] = {1.0, 1.0, 1.0, 1.0};
	GLfloat lmodel_ambient[] = {0.2, 0.2, 0.2, 1.0};
	GLfloat mat_specular[]   = {1.0, 1.0, 1.0, 1.0};
	GLfloat mat_shininess[]  = {10.0};
	glClearColor(0.0, 0.0, 0.0, 0.0);
	glShadeModel(GL_SMOOTH);
	glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE);
	glLightfv(GL_LIGHT0, GL_POSITION, light_position);
	glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
	glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);
	glLightModelfv(GL_LIGHT_MODEL_AMBIENT, lmodel_ambient);
	glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular);
	glMaterialfv(GL_FRONT, GL_SHININESS, mat_shininess);
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	glEnable(GL_COLOR_MATERIAL);
	glEnable(GL_DEPTH_TEST);
	glutKeyboardFunc(keyPressed);
	glutDisplayFunc(drawPicture);
	
	float3 eye = {0.0f, 0.0f, 2.0f*GlobeRadius};
	float near = 0.2;
	float far = 5.0*GlobeRadius;
	
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glFrustum(-0.2, 0.2, -0.2, 0.2, near, far);
	glMatrixMode(GL_MODELVIEW);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	gluLookAt(eye.x, eye.y, eye.z, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
	
	glutMainLoop();
	return 0;
}





