#include <iostream>
#include <mpi.h>
#include "input.h"
#include "inputParallel.h"
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
// Function Declaration                                                     !
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!

void setOut(double* out,double* Phi ,int which);
void setIn(double* Phi,int which ,double* In);

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
// Main Function starts here                                                !
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
void haloExchange(double* Phi, int* commMatrix, int procRank){

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
double *outN;outN = (double*) malloc(nxcGL * sizeof(double));
double *outE;outE = (double*) malloc(nycGL * sizeof(double));
double *outS;outS = (double*) malloc(nxcGL * sizeof(double));
double *outW;outW = (double*) malloc(nycGL * sizeof(double));
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
double *inN;inN = (double*) malloc(nxcGL * sizeof(double));
double *inE;inE = (double*) malloc(nycGL * sizeof(double));
double *inS;inS = (double*) malloc(nxcGL * sizeof(double));
double *inW;inW = (double*) malloc(nycGL * sizeof(double));
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
int partner;
int tag;
int nreq = 8;   // No of requests and No of status // For 2D -- 4+4 requests

MPI_Request requests[nreq];
MPI_Status status[nreq];

// Initialize requests
for( int i = 0; i<nreq; i++){
  requests[i] = MPI_REQUEST_NULL;
}
/* RECEIVING DATA TO GHOST CELLS*/
// Receive From North  Neighbor

if(commMatrix[0]==1){
  partner = procRank-nprocx;
  tag=0;
  MPI_Irecv(inN,nxcGL,MPI_DOUBLE,partner,tag,MPI_COMM_WORLD,&requests[0]);
}

// Receive From East  Neighbor

if(commMatrix[1]==1){
   partner = procRank+1;
   tag=1;
   MPI_Irecv(inE,nycGL,MPI_DOUBLE,partner,tag,MPI_COMM_WORLD,&requests[1]);
}

// Receive From South  Neighbor

if(commMatrix[2]==1){
   partner = procRank+nprocx;
   tag=2;
   MPI_Irecv(inS,nxcGL,MPI_DOUBLE,partner,tag,MPI_COMM_WORLD,&requests[2]);
}

// Receive From West  Neighbor

if(commMatrix[3]==1){
   partner = procRank-1;
   tag=3;
   MPI_Irecv(inW,nycGL,MPI_DOUBLE,partner,tag,MPI_COMM_WORLD,&requests[3]);
}

/* SENDING DATA TO GHOST CELLS */


// Send North Data to South boundary of neighbor

if(commMatrix[0]==1){
  partner = procRank-nprocx;
  tag=2;
  setOut(outN,Phi,0);
  MPI_Isend(outN,nxcGL,MPI_DOUBLE,partner,tag,MPI_COMM_WORLD,&requests[4]);
}

// Send East data to West boundary of neighbor

if(commMatrix[1]==1){
  partner = procRank+1;
  tag=3;
  setOut(outE,Phi,1);
  MPI_Isend(outE,nycGL,MPI_DOUBLE,partner,tag,MPI_COMM_WORLD,&requests[5]);
}

// Send South Data to North boundary of neighbor

if(commMatrix[2]==1){
  partner = procRank+nprocx;
  tag=0;
  setOut(outS,Phi,2);
  MPI_Isend(outS,nxcGL,MPI_DOUBLE,partner,tag,MPI_COMM_WORLD,&requests[6]);
}

// Send West Data to East boundary of neighbor

if(commMatrix[3]==1){
  partner = procRank-1;
  tag=1;
  setOut(outW,Phi,3);
  MPI_Isend(outW,nycGL,MPI_DOUBLE,partner,tag,MPI_COMM_WORLD,&requests[7]);
}
/* Wait for all communicationst to complete*/

MPI_Waitall(8,requests,status);

/* Copy Boundary values into Ghost cells, sent by neighbors into Matrix Phi */

if(commMatrix[0]==1){setIn(Phi,0,inN);}
if(commMatrix[1]==1){setIn(Phi,1,inE);}
if(commMatrix[2]==1){setIn(Phi,2,inS);}
if(commMatrix[3]==1){setIn(Phi,3,inW);}

free(outN);
free(outS);
free(outE);
free(outW);
free(inN);
free(inS);
free(inE);
free(inW);
return;
}

/* Pulls off the edge values of Phi to send to another task*/
void setOut(double* out,double* Phi ,int which){

int i;
int j;
switch(which)
{
  case 0:{  // Pull of North boundary data
  i=1; // copy the second row data
  for(j=0;j<nxcGL;j++){out[j]=Phi[i*nxcGL+j];}break;}

  case 1:{  // Pull of East boundary data
  j=nxcGL-2; // copy second last column data
  for(i=0;i<nycGL;i++){out[i]=Phi[i*nycGL+j];}break;}

  case 2:{  // Pull of South boundary data
  i=nycGL-2; //copy the second last row data
  for(j=0;j<nxcGL;j++){out[j]=Phi[i*nxcGL+j];}break;}

  case 3:{  // Pull of West boundary data
  j=1; // copy the second column data
  for(i=0;i<nycGL;i++){out[i]=Phi[i*nxcGL+j];}break;}

}

return;
}
void setIn(double* Phi,int which ,double* In){

int i;
int j;
switch(which)
{
  case 0:{ // Copy data to North Ghost cell boundary
  i=0; // copy to first row
  for(j=0;j<nxcGL;j++){Phi[i*nxcGL+j]=In[j];}break;}

  case 1:{ // Copy data to East  Ghost cell boundary
  j=nxcGL-1; //copy to last column
  for(i=0;i<nycGL;i++){Phi[i*nycGL+j]=In[i];}break;}

  case 2:{ // Copy data to South Ghost cell boundary
  i=nycGL-1;//copy to last row
  for(j=0;j<nxcGL;j++){Phi[i*nxcGL+j]=In[j];}break;}

  case 3:{ // Copy data to West  Ghost cell boundary
  j=0; //copy to first column
  for(i=0;i<nycGL;i++){Phi[i*nycGL+j]=In[i];}break;}
}

return;
}



