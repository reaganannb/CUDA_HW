// Name: Reagan Burleson
// Optimizing nBody GPU code. 
// nvcc nBodySpeedChallenge.cu -o temp -lglut -lm -lGLU -lGL

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
#define BLOCK_SIZE 1024
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
float3 *P, *V, *F;
float *M, *MInv; 
float3 *PGPU, *VGPU, *FGPU;
float *MGPU, *MInvGPU;
float GlobeRadius, Diameter, Radius;
float Damp;
dim3 BlockSize;
dim3 GridSize;

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
	int i;
	
	glClear(GL_COLOR_BUFFER_BIT);
	glClear(GL_DEPTH_BUFFER_BIT);
	
	cudaMemcpyAsync(P, PGPU, N*sizeof(float3), cudaMemcpyDeviceToHost);
	cudaErrorCheck(__FILE__, __LINE__);
	
	glColor3d(1.0,1.0,0.5);
	for(i=0; i<N; i++)
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
    	
    BlockSize.x = BLOCK_SIZE;
	BlockSize.y = 1;
	BlockSize.z = 1;
	
	GridSize.x = (N - 1)/BlockSize.x + 1; //Makes enough blocks to deal with the whole vector.
	GridSize.y = 1;
	GridSize.z = 1;
	
	Damp = 0.5;
	
	M = (float*)malloc(N*sizeof(float));
	MInv = (float*)   malloc(N*sizeof(float));
	P = (float3*)malloc(N*sizeof(float3));
	V = (float3*)malloc(N*sizeof(float3));
	
	cudaMalloc(&MGPU,N*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&MInvGPU, N*sizeof(float));
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
				if(d < Diameter)
				{
					test = 0;
					break;
				}
			}
		}
	
		V[i].x = 0.0;
		V[i].y = 0.0;
		V[i].z = 0.0;
		
		M[i] = 1.0;
		MInv[i] = 1.0f;
	}
	
	cudaMemcpyAsync(PGPU, P, N*sizeof(float3), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemcpyAsync(VGPU, V, N*sizeof(float3), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemcpyAsync(MGPU, M, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemcpy(MInvGPU, MInv, N*sizeof(float),  cudaMemcpyHostToDevice);
    cudaErrorCheck(__FILE__, __LINE__);
	
	printf("\n To start timing go to the nBody window and type s.\n");
	printf("\n To quit type q in the nBody window.\n");
}

__global__ void stepBodies(float3 *__restrict__ p,
                           float3 *__restrict__ v,
                           const float *__restrict__ m,
                           const float *__restrict__ invM,
                           float damp, float dt, float t,
                           float g, float h,
                           int n)
{
    extern __shared__ float shmem[];
    float3* sPos  = (float3*)shmem;
    float*  sMass = (float*)&sPos[blockDim.x];

    int i = threadIdx.x + blockDim.x*blockIdx.x;
    if (i >= n) return;

    float3 myPos  = p[i];
    float3 myVel  = v[i];
    float  myMass = m[i];
    float  invMass = invM[i];

    float3 acc;
    acc.x = 0.0f;
    acc.y = 0.0f;
    acc.z = 0.0f;

    float gMy = g * myMass;
    float hMy = h * myMass;

    // Loop over tiles of bodies
    for (int tile = 0; tile < n; tile += blockDim.x)
    {
        int idx = tile + threadIdx.x;
        if (idx < n)
        {
            sPos[threadIdx.x]  = p[idx];
            sMass[threadIdx.x] = m[idx];
        }
        else
        {
            sPos[threadIdx.x].x = 0.0f;
            sPos[threadIdx.x].y = 0.0f;
            sPos[threadIdx.x].z = 0.0f;
            sMass[threadIdx.x]  = 0.0f;
        }
        __syncthreads();

        int tileSize = min(blockDim.x, n - tile);
        int selfIdx  = (i >= tile && i < tile + tileSize) ? (i - tile) : -1;

        #pragma unroll 4
        for (int j = 0; j < tileSize; ++j)
        {
            if (j == selfIdx) continue;

            float mj = sMass[j];

            float dx = sPos[j].x - myPos.x;
            float dy = sPos[j].y - myPos.y;
            float dz = sPos[j].z - myPos.z;

            float d2 = dx*dx + dy*dy + dz*dz + 1e-9f; // avoid 0
            float invDist  = rsqrtf(d2);        // 1 / r
            float invDist2 = invDist * invDist; // 1 / r^2
            float invDist3 = invDist2 * invDist;
            float invDist5 = invDist3 * invDist2;

            // (G/r^2 - H/r^4)*(vec/r) = (G/r^3 - H/r^5)*vec
            float scale = (gMy * mj * invDist3) - (hMy * mj * invDist5);

            acc.x += scale * dx;
            acc.y += scale * dy;
            acc.z += scale * dz;
        }
        __syncthreads();
    }

    // Integrate: same logic as moveBodies, but using acc directly
    if(t == 0.0f)
    {
        float factor = dt * 0.5f;
        myVel.x += (acc.x - damp*myVel.x) * invMass * factor;
        myVel.y += (acc.y - damp*myVel.y) * invMass * factor;
        myVel.z += (acc.z - damp*myVel.z) * invMass * factor;
    }
    else
    {
        float factor = dt;
        myVel.x += (acc.x - damp*myVel.x) * invMass * factor;
        myVel.y += (acc.y - damp*myVel.y) * invMass * factor;
        myVel.z += (acc.z - damp*myVel.z) * invMass * factor;
    }

    myPos.x += myVel.x * dt;
    myPos.y += myVel.y * dt;
    myPos.z += myVel.z * dt;

    // Write back
    v[i] = myVel;
    p[i] = myPos;
}

void nBody()
{
	int    drawCount = 0; 
	float  t = 0.0;
	float dt = 0.0001;

	size_t shmemSize = (sizeof(float3) + sizeof(float)) * BLOCK_SIZE;

	while(t < RUN_TIME)
	{
		int steps = STEPS_PER_LAUNCH;
        if (t + steps*dt > RUN_TIME)
        {
            steps = (int)((RUN_TIME - t)/dt);
            if (steps <= 0) break;
        }

        for (int s = 0; s < steps; ++s)
        {
            stepBodies<<<GridSize, BlockSize, shmemSize>>>(PGPU, VGPU, MGPU,
                                                            MInvGPU,
                                                           Damp, dt, t,
                                                           G, H,
                                                           N);
            cudaErrorCheck(__FILE__, __LINE__);

            t += dt;
            drawCount++;
        }
		if(drawCount == DRAW_RATE) 
		{
			if(DrawFlag) 
			{	
				drawPicture();
			}
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





