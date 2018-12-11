#include <iostream>
#include <fstream>
using namespace std;
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
void printData(double* ux, double *uy, double *p,
               int col, int row){
    
ofstream myfile1;
myfile1.open ("uxVelocity.txt");
for(int i=0; i<row; ++i){
    for(int j=0; j<col; ++j){
        myfile1 << ux[i*col+j] << "\t";}
        myfile1<<endl;
        }
        myfile1.close();
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!    
ofstream myfile2;
myfile2.open ("uyVelocity.txt");
    for(int i=0; i<row; i++){
        for(int j=0; j<col; ++j){
                myfile2 << uy[i*col+j] << "\t";}
            myfile2<<endl;
        }
myfile2.close();
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!       
ofstream myfile3;
myfile3.open ("pressure.txt");
    for(int i=0; i<row; i++){
            for(int j=0; j<col; ++j){
                myfile3<<p[i*col+j] << "\t";}
            myfile3<<endl;
        }
myfile3.close();
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++! 
ofstream myfile4;
myfile4.open ("uxVertCenterVelocity.txt");
    for(int i=row-1; i>=0; i--){
        int j = col/2;
            myfile4 << ux[i*col+j] << "\t";
        myfile4<<endl;}
myfile4.close();
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
ofstream myfile5;
myfile5.open ("uyHorzCenterVelocity.txt");
    for(int j=0; j<col; j++){
        int i = row/2;
        myfile5 << uy[i*col+j] << "\t";
        myfile5<<endl;
    }
    myfile5.close();
}
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
