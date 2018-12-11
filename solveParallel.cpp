/*---------------------*- C++ 2D Incompressible FLow -*-----------------------*
|  Solves the  2D incompressible Fluid Flow in 2D geometry                    |
|  User Input is located in include/input.h                                   |
|  Subroutine name: solveParallel-> Solves in Parallel Mode                   |
*-----------------------------------------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mpi.h"

#include "input.h"
#include "inputParallel.h"
/*----------------------------------------------------------------------------*
|                    Function Declarations                                    |
*----------------------------------------------------------------------------*/
void InitializeField(double *phi, int row, int col);
void haloExchange(double* Phi, int* commMatrix, int procRank);
void updatePhysicalNorthBC(double* uxL,double* uyL,double* pL,int ProcRank);
void updatePhysicalSouthBC(double* uxL,double* uyL,double* pL,int ProcRank);
void updatePhysicalEastBC(double* uxL,double* uyL,double* pL,int ProcRank);
void updatePhysicalWestBC(double* uxL,double* uyL,double* pL,int ProcRank);
void doNavier(double* ux,double* uy,double* p,double* uxOld,double* uyOld,
                                        double* pOld, int itr, int ncGd, 
                                       int nxcGd, int nycGd);
void printData(double* uxL, double *uyL, double *pL,
               int col, int row);
void normL2(double *Phi1new, double *Phi1old,
	    double *Phi2new, double *Phi2old,
            double *Phi3new, double *Phi3old,
            double *L2Phi,   int  totCell);
/*----------------------------------------------------------------------------*
|                      Main Subroutine                                        |
*----------------------------------------------------------------------------*/
void solveParallel(int rank ,int* comm){
// For GPU computing we need to copy data to the GPU and then do the calculation
// CPU is called host and GPU is called device by convention
// Host vectors have prefix "h" and device vectors have prefix "d"
int temp = ncGL;

size_t bytes = temp*sizeof(double);

// Host vectors declaration and allocation
// Declaration of arrays for storing Primitive Variables
  double *h_ux;h_ux = (double*) malloc(bytes);
  double *h_uy;h_uy = (double*) malloc(bytes);
  double *h_p; h_p = (double*) malloc(bytes);

// storing previous time step velocity,pressure
  double *h_uxOld;h_uxOld = (double*) malloc(bytes);
  double *h_uyOld;h_uyOld = (double*) malloc(bytes);
  double *h_pOld; h_pOld  = (double*) malloc(bytes);

//Storing L2 norm 
double *L2; L2  = (double*) malloc(nvar * sizeof(double));
double *L2o;L2o = (double*) malloc(nvar * sizeof(double));

// Declaration and Initialization of variables for L2 norm calculation
double totNormUx = 1.0;
double totNormUy = 1.0;
double totNormP  = 1.0;

// Initialization of arrays for storing Primitive Variables
InitializeField(h_ux,nycGL,nxcGL);
InitializeField(h_uy,nycGL,nxcGL);
InitializeField(h_p ,nycGL,nxcGL);

InitializeField(h_uxOld,nycGL,nxcGL);
InitializeField(h_uyOld,nycGL,nxcGL);
InitializeField(h_pOld,nycGL,nxcGL);

InitializeField(L2,nvar,1);
InitializeField(L2o,nvar,1);

//--------------- Update Physical boundary ----------------------/
updatePhysicalNorthBC(h_ux,h_uy,h_p,rank);
updatePhysicalSouthBC(h_ux,h_uy,h_p,rank);
updatePhysicalEastBC(h_ux,h_uy,h_p,rank);
updatePhysicalWestBC(h_ux,h_uy,h_p,rank);
//--------------- Update Ghost layers----------------------------/
haloExchange(h_ux,comm,rank);
haloExchange(h_uy,comm,rank);
haloExchange(h_p,comm,rank);
//----------------------------------------------------------------------------!

FILE * FILE1;
FILE * FILE2;

if(rank==0){
FILE1 = fopen ("logParallel.txt" , "w");
FILE2 = fopen ("ResidualPlotting.txt" , "w");
// Write Data= logfile into file
fprintf(FILE1,"2D Navier Stokes Equation Using Finite Volume Method\n");
fprintf(FILE1,"Solving in parallel mode\n");
fprintf(FILE1,"No of Processors:\t %d \t X %d == \t %d\n",nprocx,nprocy,nproc);
fprintf(FILE1,"GRID Size:\t %d \t X %d \n",nxc,nyc);
//fprintf(FILE1,"Time Step for the simulation: %f\n",dt);
fprintf(FILE1,"Reynolds number of the simulation:%f\n",Re);
}

//Looping Variables------------------------------------------------------------!
int itr = 0;
int stop =0;

// Time Marching Loop

while (stop ==0){
itr++;
//--------------- Solve Navier Stokes equation -------------------/
doNavier(h_ux,h_uy,h_p,h_uxOld,h_uyOld,h_pOld,itr,ncGL,nxcGL,nycGL);


//--------------- Update reference Pressure ----------------------/
//refValueUpdateParallel(p,nycGL,nxcGL,nxcGL+2);

//--------------- Update Physical boundary ----------------------/
updatePhysicalNorthBC(h_ux,h_uy,h_p,rank);
updatePhysicalSouthBC(h_ux,h_uy,h_p,rank);
updatePhysicalEastBC(h_ux,h_uy,h_p,rank);
updatePhysicalWestBC(h_ux,h_uy,h_p,rank);

//--------------- Update Ghost layers----------------------------/
haloExchange(h_ux,comm,rank);
haloExchange(h_uy,comm,rank);
haloExchange(h_p,comm,rank);

//--------------------------------------------------------------------------!
// L2-Norm Calculation                                                      !
//--------------------------------------------------------------------------!
normL2(h_ux,h_uxOld,h_uy,h_uyOld,h_p,h_pOld,L2,ncGL);

MPI_Allreduce(&L2[0],&totNormUx,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
MPI_Allreduce(&L2[1],&totNormUy,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
MPI_Allreduce(&L2[2],&totNormP,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);

if(itr==1){L2o[0]=totNormUx;L2o[1]=totNormUy;L2o[2]=totNormP;}

totNormUx = sqrt(totNormUx/L2o[0]); // normalized norm wrt to initial correction
totNormUy = sqrt(totNormUy/L2o[1]);
totNormP  = sqrt(totNormP/L2o[2]);

if(rank==0){
printf("Iteration no:\t%d\t Ures: \t%.10e\t Vres: \t%.10e\t Pres: \t%.10e\t \n",itr,totNormUx,totNormUy,totNormP);
}
//--------------------------------------------------------------------------!
// Stopping Criteria                                                        !
//--------------------------------------------------------------------------!
if(MAXitr<itr){
stop = 1;
}
if((totNormUx<MAXnormux) &&(totNormUy<MAXnormuy)){
stop = 1;
}
//--------------------------------------------------------------------------!
// Writing LogFile                                                          !
//--------------------------------------------------------------------------!
if(rank==0){
fprintf(FILE1,"Iteration no:\t%d\t Ures: \t%.10e\t Vres: \t%.10e\t Pres: \t%.10e\t \n",itr,totNormUx,totNormUy,totNormP);
if(itr>0){
fprintf(FILE2,"%d\t %6.10f\t  %6.10f\t  %6.10f\t \n",itr,totNormUx,totNormUy,totNormP);
}
}
}

//-----------END OF TIME STEP ^-----------------------------!
if(rank==0){fprintf(FILE1,"Solution converged");}
if(rank==0){fclose(FILE1);fclose(FILE2);}

//--------------------------------------------------------------------------!
// Printing cell center data
// Remove the ghost cell data 
double *sendUx;sendUx = (double*) malloc(ncL * sizeof(double));
double *sendUy;sendUy = (double*) malloc(ncL * sizeof(double));
double *sendP; sendP  = (double*) malloc(ncL * sizeof(double));

for(int i=0; i<nxcL;i++){
  for(int j=0;j<nycL;j++){
    sendUx[i*nxcL+j]= h_ux[(i+1)*nxcGL+(j+1)];
    sendUy[i*nxcL+j]= h_uy[(i+1)*nxcGL+(j+1)];
    sendP[i*nxcL+j] = h_p[(i+1)*nxcGL+(j+1)];
  }
}

double *GlobalUx;GlobalUx = (double*) malloc(nc * sizeof(double));
double *GlobalUy;GlobalUy = (double*) malloc(nc * sizeof(double));
double *GlobalP; GlobalP  = (double*) malloc(nc * sizeof(double));
double *resultUx;resultUx = (double*) malloc(nc * sizeof(double));
double *resultUy;resultUy = (double*) malloc(nc * sizeof(double));
double *resultP; resultP  = (double*) malloc(nc * sizeof(double));

// Sending Results to task 0                                                !
 MPI_Gather(sendUx,ncL,MPI_DOUBLE,GlobalUx,ncL,MPI_DOUBLE,0,MPI_COMM_WORLD);
 MPI_Gather(sendUy,ncL,MPI_DOUBLE,GlobalUy,ncL,MPI_DOUBLE,0,MPI_COMM_WORLD);
 MPI_Gather(sendP ,ncL,MPI_DOUBLE,GlobalP ,ncL,MPI_DOUBLE,0,MPI_COMM_WORLD);
//--------------------------------------------------------------------------!
if(rank==0){
// we have gathered block by block into Global and is not in correct order
// Reordering of data before printing
// This step may not required if we print the varialbles along with coordinates

int index =0;
for(int k=0;k<nprocx;k++){
  for( int l=0;l<nprocy;l++){
    for(int i =k*nxcL;i<(k+1)*nxcL;i++){
      for(int j=l*nycL;j<(l+1)*nycL;j++){
        resultUx[i*nxc+j]=GlobalUx[index];
        resultUy[i*nxc+j]=GlobalUy[index];
         resultP[i*nxc+j]=GlobalP[index];
        index++;
      }
    }
  }
}
}
//--------------------------------------------------------------------------!
if(rank==0){printData(resultUx,resultUy,resultP,nxc,nyc);}
//if(rank==0){printData(h_ux,h_uy,h_p,nxcGL,nycGL);}

free(h_ux);
free(h_uy);
free(h_p);
free(h_uxOld);
free(h_uyOld);
free(h_pOld);
free(sendUx);
free(sendUy);
free(sendP);
free(L2o);
free(L2);
free(GlobalUx);
free(GlobalUy);
free(GlobalP);
free(resultUx);
free(resultUy);
free(resultP);
}
// * * * * * * * * * * END  OF SUB ROUTINE * * * * * * * * * * * * * * * * * //
