// C++ Script to simulate 2D incompressible flow
#include <iostream>
#include <algorithm>
using namespace std;
#include "input.h"
#include "inputSerial.h"

void timeStep(double* delt,double* ux,double* uy){
double umax = 1.0;  // for other cases you need to set this or write function
//double vmax = 1.0;
double C   = 0.9;  // Courant number
double dt1=1;
double dt2=1;
double dt3=1;
double delX = dx;
double dely = dy;
if((dx!=0)&(dy!=0)){ 
 dt1 = C*dx/umax;
 dt2 = C*0.25*Re/((1.0/(delX*delX))+(1.0/(dely*dely)));
 dt3 = C*4/Re;
}

double dtmin = min(dt1,min(dt2,dt3));
*delt = dtmin;
}
