/*---------------------*- C++ 2D Incompressible FLow -*-----------------------*
|  Solves the  2D incompressible Fluid Flow in 2D geometry                    |
|  User Input is input.h File                                                 |
|  This is the main file of the solver                                        |
*-----------------------------------------------------------------------------*/
#include <stdlib.h>
#include<stdio.h>
#include<math.h>
#include "mpi.h"
#include "input.h"
#include "inputSerial.h"
#include "inputParallel.h"
/*----------------------------------------------------------------------------*
|                    Function Declarations                                    |
*----------------------------------------------------------------------------*/
void solveSerial();
void mpiCheck( int procRank, int totalTasks, int totProc);
void setcomm( int procRank, int* comm, int totProc,int nchunkx,int nchunky);
void solveParallel(int processRank ,int* commMatrix);
void mesh2Dsquare(double* XX, double *YY, double *ZZ,
                  int nxcM, int nycM,int nxM, int nyM, int ncM);
/*----------------------------------------------------------------------------*
|                      Main Function                                          |
*----------------------------------------------------------------------------*/
int main(int argc, char **argv)
{
	//For Parallel Solver
	int ntasks;
	int rank;
	double wtime;
	int nfriend =4;  // for 2D we have 4 neighbors
	int comm[nfriend];

double X[nx*ny];
double Y[nx*ny];
double Z[nx*ny];

mesh2Dsquare(X,Y,Z,nxc,nyc,nx,ny,nxc*nyc);


switch(Mode){
  case 'S':
	printf(" Begin Solving in Serial Mode.\n");
    	solveSerial();                                    // Calling Serial Solver
    	break;
  case 'P':
        //MPI Initializations
        MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &ntasks);
	wtime = MPI_Wtime();
	mpiCheck( rank, ntasks, nproc);
	//Figure out Friends of each chunk
	if(rank==0){printf(" Setting up the list of neighbors.\n");}
	setcomm(rank,comm,nproc,nprocx,nprocy);
	if(rank==0){printf(" Begin Solving in Parallel Mode.\n");}
        solveParallel(rank,comm);

	// Report Wall time
	wtime=MPI_Wtime()-wtime;
        if(rank==0){
	printf("Each of  %d tasks took %6.3f seconds\n",ntasks, wtime);}
	// Terminate MPI
	MPI_Finalize();
	if(rank==0){printf("Normal End of Parallel exectution.\n");}
	break;
  default :
	printf(" Begin Solving in Serial Mode.\n");
    	solveSerial();                                    // Calling Serial Solver
    	break;
}
return 0;
}
// * * * * * * * * * * END  OF PROGRAM * * * * * * * * * * * * * * * * * * * //
