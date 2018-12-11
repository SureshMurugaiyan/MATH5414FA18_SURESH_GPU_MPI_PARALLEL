#include <iostream>
#include "input.h"
#include <cuda.h>
__global__ void eulerPredictorGPU(double* ux, double *uy,double delT,
                    double* Cnx, double *Cny,
                    double* Dnx,double* Dny,
                    int col, int row,double vol){
  // Get global thread ID
  int k = blockIdx.x*blockDim.x+threadIdx.x;
  int c = k%col;
  int r = k/col;
  if(c>0 && c<col-1 && r>0 && r<row-1){
        ux[k] = ux[k]+(delT/vol)*(-Cnx[k]+(Dnx[k])/Re);
        uy[k] = uy[k]+(delT/vol)*(-Cny[k]+(Dny[k])/Re);
    }
}


__global__ void adamPredictorGPU(double* ux, double *uy,double delT,
                      double* Cnx, double *Cny,
                      double* Dnx,double* Dny,
                      double* CnxOld, double *CnyOld,
                      double* DnxOld,double* DnyOld,
                      int col, int row,double vol){
  // Get global thread ID
  int k = blockIdx.x*blockDim.x+threadIdx.x;
  int c = k%col;
  int r = k/col;
  if(c>0 && c<col-1 && r>0 && r<row-1){

        ux[k] = ux[k]+(delT/vol)*(
                      1.5*(-Cnx[k]+(Dnx[k])/Re)
                     -0.5*(-CnxOld[k]+(DnxOld[k])/Re)
                      );
        uy[k] = uy[k]+(delT/vol)*(
                      1.5*(-Cny[k]+(Dny[k])/Re)
                     -0.5*(-CnyOld[k]+(DnyOld[k])/Re)
                      );
    
}
}


__global__ void correctorGPU(double* ux, double* uy,
               double* gradxP,double* gradyP,
               double dt, int col, int row){
  // Get global thread ID
  int k = blockIdx.x*blockDim.x+threadIdx.x;
  int c = k%col;
  int r = k/col;
  if(c>0 && c<col-1 && r>0 && r<row-1){

        ux[k] = ux[k]-dt*gradxP[k];
        uy[k] = uy[k]-dt*gradyP[k];  
   }
}


void eulerPredictor(double* ux, double *uy,double delT,
                    double* Cnx, double *Cny,
                    double* Dnx,double* Dny,
                    int col, int row,double vol){
for(int i = 1; i<(row-1); ++i){
    for(int j =1; j<(col-1); ++j){
        ux[i*col+j] = ux[i*col+j]+(delT/vol)*(-Cnx[i*col+j]+(Dnx[i*col+j])/Re);
        uy[i*col+j] = uy[i*col+j]+(delT/vol)*(-Cny[i*col+j]+(Dny[i*col+j])/Re);
    }
}

}


void adamPredictor(double* ux, double *uy,double delT,
                      double* Cnx, double *Cny,
                      double* Dnx,double* Dny,
                      double* CnxOld, double *CnyOld,
                      double* DnxOld,double* DnyOld,
                      int col, int row,double vol){
for(int i = 1; i<(row-1); ++i){
    for(int j =1; j<(col-1); ++j){
        ux[i*col+j] = ux[i*col+j]+(delT/vol)*(
                      1.5*(-Cnx[i*col+j]+(Dnx[i*col+j])/Re)
                     -0.5*(-CnxOld[i*col+j]+(DnxOld[i*col+j])/Re)
                      );
        uy[i*col+j] = uy[i*col+j]+(delT/vol)*(
                      1.5*(-Cny[i*col+j]+(Dny[i*col+j])/Re)
                     -0.5*(-CnyOld[i*col+j]+(DnyOld[i*col+j])/Re)
                      );
    }
}
}


void corrector(double* ux, double* uy,
               double* gradxP,double* gradyP,
               double dt, int col, int row){

for(int i = 1; i<(row-1); ++i){
    for(int j =1; j<(col-1); ++j){
        ux[i*col+j] = ux[i*col+j]-dt*gradxP[i*col+j];
        uy[i*col+j] = uy[i*col+j]-dt*gradyP[i*col+j];
    }
}

}


