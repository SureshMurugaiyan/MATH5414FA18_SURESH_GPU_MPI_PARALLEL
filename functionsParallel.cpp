#include "input.h"
#include "inputParallel.h"
#include <iostream>
#include <stdlib.h> // Input output system
#include <stdio.h> // File System
#include <math.h>
using namespace std;

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
// Update Physical Boundary conditions for Parallel solver-North Boundary   !
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
void updatePhysicalNorthBC(double* uxL,double* uyL,double* pL,int ProcRank){
//if(ProcRank==0||ProcRank==1){
if(ProcRank<nprocx){
    for (int i = 0; i<nxcGL; i++){
        uxL[i]= 1;
        uyL[i]= 0;
        pL[i]  = pL[i+nxcGL];
    }
}
}
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
// Update Physical Boundary conditions for Parallel solver-South Boundary   !
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
void updatePhysicalSouthBC(double* uxL,double* uyL,double* pL,int ProcRank){
//if(ProcRank==2||ProcRank==3){
if(ProcRank>(nprocx*(nprocy-1)-1)){
    for (int i = ncGL-nxcGL; i<ncGL; i++){
        uxL[i]= 0;
        uyL[i]= 0;
        pL[i]= pL[i-nxcGL];
    }
}
}
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
// Update Physical Boundary conditions for Parallel solver-East  Boundary   !
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
void  updatePhysicalEastBC(double* uxL,double* uyL,double* pL,int ProcRank){
//if(ProcRank==1||ProcRank==3){
int temp = nprocx;
if(((ProcRank+1)%temp)==0){
    for (int i = nxcGL-1; i<ncGL; i=(i+nxcGL)){
        uxL[i]=0;
        uyL[i]=0;
        pL[i]  = pL[i-1];
    }
}
}
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
// Update Physical Boundary conditions for Parallel solver-West  Boundary   !
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
void  updatePhysicalWestBC(double* uxL,double* uyL,double* pL,int ProcRank){

//if(ProcRank==0||ProcRank==2){
int temp = nprocx;
if ( ( ProcRank % temp ) == 0 ){
    for (int i = 0; i<ncGL; i=(i+nxcGL)){
        uxL[i]= 0;
        uyL[i]= 0;
        pL[i] = pL[i+1];
    }
}
}
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
