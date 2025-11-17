// Name: Reagan Burleson
// Creating a n = whatever from an n <= 1024 nBody GPU code. 
// nvcc 19LargeNBody.cu -o temp -lglut -lm -lGLU -lGL

/*
 What to do:
 This is some lean nBody code that runs on the GPU. But the number of bodies it can simulated is limited to 1024
 so it can run on one block. Extend this code so it can simulation as many bodies as the user wants (within reason).
 Keep the same general format. But you will need to change a few major things in the code.
 Note: The code takes two arguments as inputs:
 1. The number of bodies to simulate.
 2. Whether to draw sub-arrangements of the bodies during the simulation (1), or only the first and last arrangements (0).
*/

/*
 Purpose:
 To be learn how to extend an Nbody simulation from one block to many blocks.
*/

// Include files
#include <GL/glut.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

// Defines
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

// Globals
int N, DrawFlag;
float3 *P, *V, *F;
float *M; 
float3 *PGPU, *VGPU, *FGPU;
float *MGPU;
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
__global__ void leapFrog(float3 *, float3 *, float3 *, float *, float, float, float, float, float, int);
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
    	
    BlockSize.x = 256;
    BlockSize.y = 1;
    BlockSize.z = 1;

    GridSize.x = (N + BlockSize.x - 1) / BlockSize.x;
    GridSize.y = 1;
    GridSize.z = 1;

    	
    	Damp = 0.5;
    	
    	M = (float*)malloc(N*sizeof(float));
    	P = (float3*)malloc(N*sizeof(float3));
    	V = (float3*)malloc(N*sizeof(float3));
    	F = (float3*)malloc(N*sizeof(float3));
    	
    	cudaMalloc(&MGPU,N*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&PGPU,N*sizeof(float3));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&VGPU,N*sizeof(float3));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&FGPU,N*sizeof(float3));
	cudaErrorCheck(__FILE__, __LINE__);
    	
	
	Diameter = pow(H/G, 1.0/(LJQ - LJP)); // This is the value where the force is zero for the L-J type force.
	Radius = Diameter/2.0;
	
	// Using the radius of a body and a 68% packing ratio to find the radius of a global sphere that should hold all the bodies.
	// Then we double this radius just so we can get all the bodies setup with no problems. 
	float totalVolume = float(N)*(4.0/3.0)*PI*Radius*Radius*Radius;
	totalVolume /= 0.68;
	float totalRadius = pow(3.0*totalVolume/(4.0*PI), 1.0/3.0);
	GlobeRadius = 2.0*totalRadius;
	
	// Randomly setting these bodies in the glaobal sphere and setting the initial velosity, inotial force, and mass.
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
		
		F[i].x = 0.0;
		F[i].y = 0.0;
		F[i].z = 0.0;
		
		M[i] = 1.0;
	}
	
	cudaMemcpyAsync(PGPU, P, N*sizeof(float3), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemcpyAsync(VGPU, V, N*sizeof(float3), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemcpyAsync(FGPU, F, N*sizeof(float3), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemcpyAsync(MGPU, M, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	
	printf("\n To start timing type s.\n");
}

__global__ void leapFrog(float3 *p, float3 *v, float3 *f, float *m,
                         float g, float h, float damp, float dt, float t, int n)
{
    extern __shared__ float4 tile[]; //size = blockDim.x

    const int tid   = threadIdx.x;
    const int bdim  = blockDim.x;
    const int gdim  = gridDim.x;
    int i = blockIdx.x * bdim + tid;

    //stride over i for very large N
    for (; i < n; i += bdim * gdim)
    {
        float3 Pi = p[i];
        float3 Vi = v[i];
        const float  Mi = m[i];

        //accumulate force in registers
        float3 Fi; Fi.x = 0.f; Fi.y = 0.f; Fi.z = 0.f;

        //tile over j
        for (int base = 0; base < n; base += bdim)
        {
            int j = base + tid;

            //coalesced load of tile
            float4 val = make_float4(0.f, 0.f, 0.f, 0.f);
            if (j < n) {
                val.x = p[j].x; val.y = p[j].y; val.z = p[j].z; val.w = m[j];
            }
            tile[tid] = val;
            __syncthreads();

            const int limit = min(bdim, n - base);

            //compute interactions with tile
            #pragma unroll 32
            for (int k = 0; k < limit; ++k)
            {
                int idx = base + k;
                if (idx == i) continue;

                float dx = tile[k].x - Pi.x;
                float dy = tile[k].y - Pi.y;
                float dz = tile[k].z - Pi.z;

                //soften to avoid singularities and use fast reciprocal sqrt
                const float soft = 1e-6f;
                float r2   = dx*dx + dy*dy + dz*dz + soft;
                float invR = rsqrtf(r2);           // 1/r
                float invR2 = invR * invR;         // 1/r^2
                float invR4 = invR2 * invR2;       // 1/r^4

                //g*Mi*Mj / r^2 - h*Mi*Mj / r^4
                float mm = Mi * tile[k].w;
                float common = (g * mm) * invR2 - (h * mm) * invR4;

                Fi.x += common * dx * invR;
                Fi.y += common * dy * invR;
                Fi.z += common * dz * invR;
            }
            __syncthreads();
        }

        //write fi to glob so its visible to host
        f[i] = Fi;

        //damped velocity update
        float invMi = 1.0f / Mi;
        if (t == 0.0f) {
            Vi.x += ((Fi.x - damp*Vi.x) * invMi) * (dt * 0.5f);
            Vi.y += ((Fi.y - damp*Vi.y) * invMi) * (dt * 0.5f);
            Vi.z += ((Fi.z - damp*Vi.z) * invMi) * (dt * 0.5f);
        } else {
            Vi.x += ((Fi.x - damp*Vi.x) * invMi) * dt;
            Vi.y += ((Fi.y - damp*Vi.y) * invMi) * dt;
            Vi.z += ((Fi.z - damp*Vi.z) * invMi) * dt;
        }

        //integrate position
        Pi.x += Vi.x * dt;
        Pi.y += Vi.y * dt;
        Pi.z += Vi.z * dt;

        //write back once
        v[i] = Vi;
        p[i] = Pi;
    }
}


void nBody()
{
    int    drawCount = 0;
    float  t = 0.0f;
    float  dt = 0.0001f;

    //dynamic shared memory per block, one float4 per thread
    size_t shmem = BlockSize.x * sizeof(float4);

    while (t < RUN_TIME)
    {
        leapFrog<<<GridSize, BlockSize, shmem>>>(PGPU, VGPU, FGPU, MGPU, G, H, Damp, dt, t, N);

        //draw at cadence
        if (drawCount == DRAW_RATE) {
            if (DrawFlag) {
                cudaDeviceSynchronize();
                drawPicture();
            }
            drawCount = 0;
        }

        t += dt;
        drawCount++;
    }

    cudaDeviceSynchronize();
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
	glutCreateWindow("nBody Test");
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





