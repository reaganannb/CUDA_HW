// Name: Reagan Burleson
// Two body problem
// nvcc 17TwoBodyToNBodyCPU.cu -o temp -lglut -lGLU -lGL
//To stop hit "control c" in the window you launched it from.

/*
 What to do:
 This is some crude code that moves two bodies around in a box, attracted by gravity and 
 repelled when they hit each other. Take this from a two-body problem to an N-body problem, where 
 NUMBER_OF_SPHERES is a #define that you can change. Also clean it up a bit so it is more user friendly.
*/

/*
 Purpose:
 To learn about Nbody code.
*/

// Include files
#include <GL/freeglut.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Defines
#define NUMBER_OF_SPHERES 5
#define XWindowSize 1000
#define YWindowSize 1000
#define STOP_TIME 10000.0
#define DT        0.0001
#define GRAVITY 0.1 
#define MASS 10.0  	
#define DIAMETER 1.0
#define SPHERE_PUSH_BACK_STRENGTH 50.0
#define PUSH_BACK_REDUCTION 0.1
#define DAMP 0.01
#define DRAW 100
#define LENGTH_OF_BOX 6.0
#define MAX_VELOCITY 5.0

// Globals
const float XMax = (LENGTH_OF_BOX/2.0);
const float YMax = (LENGTH_OF_BOX/2.0);
const float ZMax = (LENGTH_OF_BOX/2.0);
const float XMin = -(LENGTH_OF_BOX/2.0);
const float YMin = -(LENGTH_OF_BOX/2.0);
const float ZMin = -(LENGTH_OF_BOX/2.0);

static float px[NUMBER_OF_SPHERES], py[NUMBER_OF_SPHERES], pz[NUMBER_OF_SPHERES];
static float vx[NUMBER_OF_SPHERES], vy[NUMBER_OF_SPHERES], vz[NUMBER_OF_SPHERES];
static float fx[NUMBER_OF_SPHERES], fy[NUMBER_OF_SPHERES], fz[NUMBER_OF_SPHERES];
static float massArr[NUMBER_OF_SPHERES];

// Function prototypes
void set_initail_conditions();
void Drawwirebox();
void draw_picture();
void keep_in_box();
void get_forces();
void move_bodies(float);
void nbody();
void Display(void);
void reshape(int, int);
int main(int, char**);

//helpers
static inline float frand01(){ return (float)rand() / (float)RAND_MAX; }
static inline float frandSym(float mag){ return (2.0f*frand01() - 1.0f)*mag; }

void set_initail_conditions()
{ 
	time_t t;
	srand((unsigned) time(&t));
	
	const float halfBoxLength = (LENGTH_OF_BOX - DIAMETER)/2.0f;

	//random positions and velocities, simple rejection for overlaps
	for(int i=0;i<NUMBER_OF_SPHERES;i++){
		//place avoiding immediate overlap with previously placed
		int placed = 0;
		for(int tries=0; tries<5000 && !placed; ++tries){
			float x = frandSym(halfBoxLength);
			float y = frandSym(halfBoxLength);
			float z = frandSym(halfBoxLength);
			int ok = 1;
			for(int j=0;j<i;j++){
				float dx = x - px[j];
				float dy = y - py[j];
				float dz = z - pz[j];
				float sep = sqrtf(dx*dx + dy*dy + dz*dz);
				if(sep < DIAMETER*1.01f){ ok = 0; break; }
			}
			if(ok){
				px[i] = x; py[i] = y; pz[i] = z;
				placed = 1;
			}
		}
		if(!placed){
			//fallback if crowded; allow placement anyway
			px[i] = frandSym(halfBoxLength);
			py[i] = frandSym(halfBoxLength);
			pz[i] = frandSym(halfBoxLength);
		}

		vx[i] = frandSym(MAX_VELOCITY);
		vy[i] = frandSym(MAX_VELOCITY);
		vz[i] = frandSym(MAX_VELOCITY);

		massArr[i] = 1.0f; //same as original mass1/mass2 spirit
	}

	//zero forces
	for(int i=0;i<NUMBER_OF_SPHERES;i++){ fx[i]=fy[i]=fz[i]=0.0f; }
}

void Drawwirebox()
{		
	glColor3f (5.0,1.0,1.0);
	glBegin(GL_LINE_STRIP);
		glVertex3f(XMax,YMax,ZMax);
		glVertex3f(XMax,YMax,ZMin);	
		glVertex3f(XMax,YMin,ZMin);
		glVertex3f(XMax,YMin,ZMax);
		glVertex3f(XMax,YMax,ZMax);
		
		glVertex3f(XMin,YMax,ZMax);
		
		glVertex3f(XMin,YMax,ZMax);
		glVertex3f(XMin,YMax,ZMin);	
		glVertex3f(XMin,YMin,ZMin);
		glVertex3f(XMin,YMin,ZMax);
		glVertex3f(XMin,YMax,ZMax);	
	glEnd();
	
	glBegin(GL_LINES);
		glVertex3f(XMin,YMin,ZMax);
		glVertex3f(XMax,YMin,ZMax);		
	glEnd();
	
	glBegin(GL_LINES);
		glVertex3f(XMin,YMin,ZMin);
		glVertex3f(XMax,YMin,ZMin);		
	glEnd();
	
	glBegin(GL_LINES);
		glVertex3f(XMin,YMax,ZMin);
		glVertex3f(XMax,YMax,ZMin);		
	glEnd();
	
}

void draw_picture()
{
	float radius = DIAMETER/2.0;
	glClear(GL_COLOR_BUFFER_BIT);
	glClear(GL_DEPTH_BUFFER_BIT);
	
	Drawwirebox();
	
	for(int i=0;i<NUMBER_OF_SPHERES;i++){
		// just two colors alternating to stay close to original look
		if(i&1) glColor3d(1.0,0.5,1.0);
		else    glColor3d(0.0,0.5,0.0);

		glPushMatrix();
		glTranslatef(px[i], py[i], pz[i]);
		glutSolidSphere(radius,20,20);
		glPopMatrix();
	}
	
	glutSwapBuffers();
}

void keep_in_box()
{
	float halfBoxLength = (LENGTH_OF_BOX - DIAMETER)/2.0;
	
	for(int i=0;i<NUMBER_OF_SPHERES;i++){
		if(px[i] > halfBoxLength){
			px[i] = 2.0f*halfBoxLength - px[i];
			vx[i] = - vx[i];
		}
		else if(px[i] < -halfBoxLength){
			px[i] = -2.0f*halfBoxLength - px[i];
			vx[i] = - vx[i];
		}

		if(py[i] > halfBoxLength){
			py[i] = 2.0f*halfBoxLength - py[i];
			vy[i] = - vy[i];
		}
		else if(py[i] < -halfBoxLength){
			py[i] = -2.0f*halfBoxLength - py[i];
			vy[i] = - vy[i];
		}

		if(pz[i] > halfBoxLength){
			pz[i] = 2.0f*halfBoxLength - pz[i];
			vz[i] = - vz[i];
		}
		else if(pz[i] < -halfBoxLength){
			pz[i] = -2.0f*halfBoxLength - pz[i];
			vz[i] = - vz[i];
		}
	}
}

void get_forces()
{
	for(int i=0;i<NUMBER_OF_SPHERES;i++){ fx[i]=fy[i]=fz[i]=0.0f; }

	const float eps = 1e-6f;
	for(int i=0;i<NUMBER_OF_SPHERES;i++){
		for(int j=i+1;j<NUMBER_OF_SPHERES;j++){
			float dx = px[j] - px[i];
			float dy = py[j] - py[i];
			float dz = pz[j] - pz[i];

			float r2 = dx*dx + dy*dy + dz*dz + eps;
			float r  = sqrtf(r2);

			float forceMag = (massArr[i]*massArr[j]*GRAVITY)/r2;

			//soft push-back when overlapping
			if (r < DIAMETER){
				float dvx = vx[j] - vx[i];
				float dvy = vy[j] - vy[i];
				float dvz = vz[j] - vz[i];
				float inout = dx*dvx + dy*dvy + dz*dvz;
				if(inout <= 0.0f){
					forceMag +=  SPHERE_PUSH_BACK_STRENGTH*(r - DIAMETER);
				} else {
					forceMag +=  PUSH_BACK_REDUCTION*SPHERE_PUSH_BACK_STRENGTH*(r - DIAMETER);
				}
			}

			//apply along the line of centers
			float invr = 1.0f / (r + eps);
			float fx_ij = forceMag * dx * invr;
			float fy_ij = forceMag * dy * invr;
			float fz_ij = forceMag * dz * invr;

			fx[i] +=  fx_ij; fy[i] +=  fy_ij; fz[i] +=  fz_ij;
			fx[j] += -fx_ij; fy[j] += -fy_ij; fz[j] += -fz_ij;
		}
	}
}

void move_bodies(float time)
{
	if(time == 0.0f){
		for(int i=0;i<NUMBER_OF_SPHERES;i++){
			vx[i] += 0.5f*DT*(fx[i] - DAMP*vx[i])/massArr[i];
			vy[i] += 0.5f*DT*(fy[i] - DAMP*vy[i])/massArr[i];
			vz[i] += 0.5f*DT*(fz[i] - DAMP*vz[i])/massArr[i];
		}
	} else {
		for(int i=0;i<NUMBER_OF_SPHERES;i++){
			vx[i] += DT*(fx[i] - DAMP*vx[i])/massArr[i];
			vy[i] += DT*(fy[i] - DAMP*vy[i])/massArr[i];
			vz[i] += DT*(fz[i] - DAMP*vz[i])/massArr[i];
		}
	}

	for(int i=0;i<NUMBER_OF_SPHERES;i++){
		px[i] += DT*vx[i];
		py[i] += DT*vy[i];
		pz[i] += DT*vz[i];
	}

	keep_in_box();
}

void nbody()
{	
	int    tdraw = 0;
	float  time = 0.0;

	set_initail_conditions();
	
	draw_picture();
	
	while(time < STOP_TIME)
	{
		get_forces();
	
		move_bodies(time);
	
		tdraw++;
		if(tdraw == DRAW) 
		{
			draw_picture(); 
			tdraw = 0;
		}
		
		time += DT;
	}
	printf("\n DONE \n");
	glutLeaveMainLoop();  
	return;               

}

void Display(void)
{
	gluLookAt(0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
	glClear(GL_COLOR_BUFFER_BIT);
	glClear(GL_DEPTH_BUFFER_BIT);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glutSwapBuffers();
	glFlush();
	nbody();
}

void reshape(int w, int h)
{
	glViewport(0, 0, (GLsizei) w, (GLsizei) h);

	glMatrixMode(GL_PROJECTION);

	glLoadIdentity();

	glFrustum(-0.2, 0.2, -0.2, 0.2, 0.2, 50.0);

	glMatrixMode(GL_MODELVIEW);
}

int main(int argc, char** argv)
{
	glutInit(&argc,argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_DEPTH | GLUT_RGB);
	glutInitWindowSize(XWindowSize,YWindowSize);
	glutInitWindowPosition(0,0);
	glutCreateWindow("2 Body 3D");
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
	glutDisplayFunc(Display);
	glutReshapeFunc(reshape);
	glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);
	glutMainLoop();
	return 0;
}


