#include<iostream>
#include<mpi.h>

void mpiCheck( int procRank, int totalTasks, int totProc){
if(procRank==0)
{
  printf("\n");
  printf("Solving 2D Navier Stokes Eqn in Parallel Mode:\n");
}

if(totalTasks!=totProc)
{
  if(procRank==0)
  {
    printf("\n Fatal error!\n");
    printf("No process should be set to %i!\n",totProc);
  }
  MPI_Finalize();
  exit(1);
}

if(procRank==0)
{
  printf("\n");
  printf("MPI has been set up.\n");
}
}

void setcomm( int procRank, int* comm, int totProc,int nchunkx,int nchunky){
  /* Determines the active communication direction
   * For 4 processor
   *    ---------
   *   | 0  |  1 |
   *    ----+----
   *   | 2  |  3 |
   *    ---------
   *    Here Processor 0 will communicate with 1 and 2
   *    COMM array will be { 0,1,1,0}
   */

// Start by assuming all neighbors exist
for(int i = 0;i<totProc;i++){comm[i]=1;}

// Check for TOP Neighbor ? No for First row
if(procRank<nchunkx){comm[0]=0;}

// Check for RIGHT neighbor? No for last column
if((procRank+1)%nchunkx==0){comm[1]=0;}

// Check for DOWN neighbor? No for last row
if(procRank>(nchunkx*(nchunky-1)-1)){comm[2]=0;}

// check for LEFT neighbor? No for first column
if((procRank%nchunkx)==0){comm[3]=0;}

return;
}
