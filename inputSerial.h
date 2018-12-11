#include "input.h"
#define nx   (nxc+1)      // Number of vertices in North/South Boundary
#define ny   (nyc+1)      // Number of vertices in East/West   Boundary
#define nv   (nx*ny)      // Number of vertices in total


#define nxcG (nxc+2)       // Number of cells in North/South Boundary including 1 layer of Ghost cells
#define nycG (nyc+2)       // Number of cells in East/West   Boundary including 1 layer of Ghost cells
#define ncG  (nxcG)*(nycG) // Number of cells in Total including 1 layer of Ghost cells


#define nxG   (nx+2)       // Number of vertices in North/South Boundary including 1 layer of Ghost cells
#define nyG   (ny+2)       // Number of vertices in East/West   Boundary including 1 layer of Ghost cells
#define nvG   (nxG*nyG)    // Number of vertices in total including 1 layer of Ghost cells

#define nxHfc  (nx-1)       // Number of face center in North/South Boundary along Horizontal co-ordinate lines
#define nyHfc  ny           // Number of face center in East/West   Boundary along Horizontal co-ordinate lines
#define nHfc   (nxHfc*nyHfc)// Number of face center in Total along Horizontal co-ordinate lines 

#define nxVfc  nx           // Number of face center in North/South Boundary along Vertical co-ordinate lines
#define nyVfc  (ny-1)       // Number of face center in East/West   Boundary along Vertical co-ordinate lines
#define nVfc   (nxVfc*nyVfc)// Number of face center in Total along Vertical co-ordinate lines


#define nxHfcG  (nxHfc+2)       // No of fcenter in North/South  alg Horz codnate lines + 1 layr of Ghost cells
#define nyHfcG  (nyHfc+2)       // No of fcenter in East/West   alg Horz codnate lines + 1 layr of Ghost cells
#define nHfcG   (nxHfcG*nyHfcG) // No of fcenter in Total alg Horz codnate lines + 1 layr of Ghost cells 

#define nxVfcG  (nxVfc+2)        // No of fcenter in North/South alg Vert codnate lines + 1 layr of Ghost cells
#define nyVfcG  (nyVfc+2)        // No of fcenter in East/West alg Vert codnate lines + 1 layr of Ghost cells
#define nVfcG   (nxVfcG*nyVfcG)  // No of fcenter in Total alg Vert codnate lines + 1 layr of Ghost cells

#define dx (1.0/nxc)
#define dy (1.0/nyc)
#define dV (dx*dy)

