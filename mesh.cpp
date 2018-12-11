// C++ Script to simulate 2D incompressible flow
#include <iostream>
#include <math.h>
#include <fstream>

using namespace std;

void mesh2Dsquare(double* X, double *Y, double *Z,
                  int nxc, int nyc,int nx, int ny, int nc){
// North boundary
const double xmin_n = -0.5;  // left  co-ordinate
const double ymin_n =  0.5;  // left  co-ordinate
const double xmax_n =  0.5;  // right co-ordinate
const double ymax_n =  0.5;  // right co-ordinate

//South boundary
const double xmin_s = -0.5;  // left  co-ordinate
const double ymin_s = -0.5;  // left  co-ordinate
const double xmax_s =  0.5;  // right co-ordinate
const double ymax_s = -0.5;  // right co-ordinate

//East boundary
const double xmin_e = xmax_s; // left  co-ordinate
const double ymin_e = ymax_s; // left  co-ordinate
const double xmax_e = xmax_n; // right co-ordinate
const double ymax_e = ymax_n; // right co-ordinate

//West boundary
const double xmin_w = xmin_s; // left  co-ordinate
const double ymin_w = ymin_s; // left  co-ordinate
const double xmax_w = xmin_n; // right co-ordinate
const double ymax_w = ymin_n; // right co-ordinate

//Length of the domain
//const double length_n = sqrt(pow((xmin_n -xmax_n),2)+pow((ymin_n-ymax_n),2));
//const double length_s = sqrt(pow((xmin_s -xmax_s),2)+pow((ymin_s-ymax_s),2));
//const double length_e = sqrt(pow((xmin_e -xmax_e),2)+pow((ymin_e-ymax_e),2));
//const double length_w = sqrt(pow((xmin_w -xmax_w),2)+pow((ymin_w-ymax_w),2));

//Evaluation of boundary lines

double x_n[nx];   // number of grid points
double x_s[nx];   // grid points = cells+1
double y_n[nx];
double y_s[nx];
double x_e[ny];
double x_w[ny];
double y_e[ny];
double y_w[ny];

for (int i = 0; i<nx; i++)
{
  x_n[i]=xmin_n+((xmax_n-xmin_n)/(nx-1))*i;   //north boundary grid points
  y_n[i]=ymin_n+((ymax_n-ymin_n)/(nx-1))*i;   // arranging from left to right
  x_s[i]=xmin_s+((xmax_s-xmin_s)/(nx-1))*i;   //south boundary grid points
  y_s[i]=ymin_s+((ymax_s-ymin_s)/(nx-1))*i;
}

for (int i = 0; i<ny; i++)
{
  x_e[i]=xmax_e-((xmax_e-xmin_e)/(ny-1))*i;   //east boundary grid points
  y_e[i]=ymax_e-((ymax_e-ymin_e)/(ny-1))*i;   //arranging from top to bottom
  x_w[i]=xmax_w-((xmax_w-xmin_w)/(ny-1))*i;   //west boundary grid points
  y_w[i]=ymax_w-((ymax_w-ymin_w)/(ny-1))*i;
}
// Initializing matrix with grid points
double x[ny][nx]; // x co-ordinates of grid
double y[ny][nx]; // y co-ordinates of grid

//Assigning boundary points in Global 2D matrix of Grid
//North and South boundary

for (int i = 0; i<nx; i++)
{
  x[0][i]    = x_n[i]; //North boundary
  y[0][i]    = y_n[i];
  x[ny-1][i] = x_s[i]; //South boundary
  y[ny-1][i] = y_s[i];
}

//East and west boundary

for (int i = 0; i<ny; i++)
{
  x[i][0]    = x_w[i]; //West boundary
  y[i][0]    = y_w[i];
  x[i][nx-1] = x_e[i]; //East boundary
  y[i][nx-1] = y_e[i];
}

// Transfinite Interpolation to find inner points

/* Computational domain is unit square
coordinate lines are Psi and Eta
Setting Psi = s ; and Eta = t
Setting dPsi = dX ; dEta = dY */
double s,t,dX,dY;

dX = 1/(nx-1);   // Numerator is unity => unit square
dY = 1/(ny-1);   // ,,

// Interpolation of inner points by TFI

for (int j = 1; j<nx-1; j++)
 {
   s = j*dX;
   for (int i = 1; i<ny-1; i++)
     {
       t = i*dY;
       x[j][i]= (1-t)*x[ny-1][i]+t*x[0][i]+(1-s)*x[j][0]+s*x[j][nx-1]\
                   -(s*t*x[0][nx-1]+s*(1-t)*x[ny-1][nx-1]\
                   +t*(1-s)*x[0][0]+(1-s)*(1-t)*x[ny-1][0]);
       y[j][i]= (1-t)*y[ny-1][i]+t*y[0][i]+(1-s)*y[j][0]+s*y[j][nx-1]\
                   -(s*t*y[0][nx-1]+s*(1-t)*y[ny-1][nx-1]\
                   +t*(1-s)*y[0][0]+(1-s)*(1-t)*y[ny-1][0]);
     }
     cout << endl;
  }

// Calculation of cell face centers
double x_hfc[ny][nx-1];   //x-co-ordinates of horizontal lines
double y_hfc[ny][nx-1];   //y-co-ordinates of horizontal lines

double x_vfc[ny-1][nx];   //x-co-ordinates of vertical lines
double y_vfc[ny-1][nx];   //y-co-ordinates of vertical lines

double x_c[ny-1][nx-1];   //x-co-ordinates of cell centers
double y_c[ny-1][nx-1];   //y-co-ordinates of cell centers

// Calculation of horizontal face centers
// center of horizontal grid lines

for (int j = 0; j<ny; j++)
 {
   for (int i = 0; i<nx-1; i++)
    {
       x_hfc[j][i]=0.5*(x[j][i]+x[j][i+1]);
       y_hfc[j][i]=0.5*(y[j][i]+y[j][i+1]);
    }
  }

// Calculation of vertical face centers
// center of vertical grid lines

for (int j = 0; j<ny-1; j++)
 {
   for (int i = 0; i<nx; i++)
    {
       x_vfc[j][i]=0.5*(x[j][i]+x[j+1][i]);
       y_vfc[j][i]=0.5*(y[j][i]+y[j+1][i]);
    }
  }


// Calculation of cell centers

for (int j = 0; j<ny-1; j++)
 {
   for (int i = 0; i<nx-1; i++)
    {
       x_c[j][i]=0.25*(x_hfc[j][i]+x_hfc[j+1][i]+x_vfc[j][i]+x_vfc[j][i+1]);
       y_c[j][i]=0.25*(y_hfc[j][i]+y_hfc[j+1][i]+y_vfc[j][i]+y_vfc[j][i+1]);
    }
  }

/* Variables*/
//double dx = length_n/nxc ;
//double dy = length_e/nyc;

//CREATING MESH FILE

    ofstream myfile1;
    myfile1.open ("xVertices.txt");
    //myfile1 << "Writing this to a file.\n";
    for(int i=0; i<ny; i++)
    {
        for(int j=0; j<nx; ++j)
        {
            myfile1 << x[i][j] << "\t";
        }
        myfile1<<endl;
    }
    myfile1.close();

    ofstream myfile2;
    myfile2.open ("yVertices.txt");
    //myfile1 << "Writing this to a file.\n";
    for(int i=0; i<ny; i++)
    {
        for(int j=0; j<nx; ++j)
        {
            myfile2 << y[i][j] << "\t";
        }
        myfile2<<endl;
    }
    myfile2.close();

    ofstream myfile3;
    myfile3.open ("xCellcenter.txt");
    //myfile1 << "Writing this to a file.\n";
    for(int i=0; i<nxc; i++)
    {
        for(int j=0; j<nxc; j++)
        {
            myfile3<<x_c[i][j] << "\t";
        }
        myfile3<<endl;
    }
    myfile3.close();

    ofstream myfile4;
    myfile4.open ("yCellcenter.txt");
    //myfile1 << "Writing this to a file.\n";
    for(int i=0; i<nyc; i++)
    {
        for(int j=0; j<nxc; ++j)
        {
            myfile4<<y_c[i][j] << "\t";
        }
        myfile4<<endl;
    }
    myfile4.close();


    ofstream myfile5;
    myfile5.open ("Mesh.txt");
    myfile5 << "$MeshFile" << endl;
    myfile5<< "Vertex" <<"\t"<< "X"<<"\t"<<"Y"<<"Z"<<endl;
    for(int i=0; i<ny; i++){
        for(int j=0; j<nx; j++){
            X[i*nx+j]= x[i][j];
            Y[i*nx+j]= y[i][j];
            myfile5<< (i*nx+j) <<"\t"<< X[i*nx+j]<<"\t"<<Y[i*nx+j]<<"0"<<endl;

        }
    }





//END OF PROGRAM
}
