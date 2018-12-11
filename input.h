//USER INPUT
#define Mode 'P'         // Choice S == Serial Mode and Choice P == Parallel mode
#define nxc 64            // Number of cells in North and South Boundary
#define nyc 64             // Number of cells in East  and West  Boundary
#define Re  100.0           // Number of cells in East  and West  Boundary
#define pRefCell 1          // Pressure Reference cell // Assume first inner cell
#define MAXitr 10000         // Maximum number of Main Loop iterations// Time marching iterations
#define MAXnormux (1e-8)     // Desired normalized L2 norm for U-Velocity
#define MAXnormuy (1e-8)     // Desired normalized L2 norm for V-Velocity
#define MAXnormp (1e-8)      // Desired normalized L2 norm for Pressure
#define MAXitrPr 2000        // Maximum number of inner Pressure Loop iterations-Poisson solver

//USER INPUT FOR PARALLEL
#define nxcL 32  // number of cells in north and south of each chunk --- chunknxc
#define nycL 32  // number of cells in each and west boundary of each chunk------chunknyc

//VARIABLES
#define nvar 3         // u,v,p // dont change it // this in only 2D solver

#define dx (1.0/nxc)
#define dy (1.0/nyc)
#define dV (dx*dy)
