#define nc  nxc*nyc  // total number of cells
#define ncL nxcL*nycL
#define nxL   (nxcL+1)      // Number of vertices in North/South Boundary
#define nyL   (nycL+1)      // Number of vertices in East/West   Boundary
#define nvL   (nxL*nyL)      // Number of vertices in total

#define nprocx nxc/nxcL  // number of chunks divided along row ---- nchunkx
#define nprocy nyc/nycL  // number of chunks divided along column---- nchunky
#define nproc nprocx*nprocy // total number of processors or taks = no of subdomains

#define nxcGL (nxcL+2)
#define nycGL (nycL+2)
#define ncGL (nxcGL)*(nycGL)

#define nxGL   (nxL+2)       // Number of vertices in North/South Boundary including 1 layer of Ghost cells
#define nyGL   (nyL+2)       // Number of vertices in East/West   Boundary including 1 layer of Ghost cells
#define nvGL   (nxGL*nyGL)    // Number of vertices in total including 1 layer of Ghost cells

#define nxHfcL  (nxL-1)       // Number of face center in North/South Boundary along Horizontal co-ordinate lines
#define nyHfcL  nyL           // Number of face center in East/West   Boundary along Horizontal co-ordinate lines
#define nHfcL   (nxHfcL*nyHfcL)// Number of face center in Total along Horizontal co-ordinate lines 

#define nxVfcL  nxL           // Number of face center in North/South Boundary along Vertical co-ordinate lines
#define nyVfcL  (nyL-1)       // Number of face center in East/West   Boundary along Vertical co-ordinate lines
#define nVfcL   (nxVfcL*nyVfcL)// Number of face center in Total along Vertical co-ordinate lines

#define nxHfcGL  (nxHfcL+2)       // No of facecenter in North/South  alg Horz codnate lines + 1 layer of Ghost cells
#define nyHfcGL  (nyHfcL+2)       // No of facecenter in East/West   alg Horz codnate lines + 1 layer of Ghost cells
#define nHfcGL   (nxHfcGL*nyHfcGL) // No of facecenter in Total alg Horz codnate lines + 1 layer of Ghost cells 

#define nxVfcGL  (nxVfcL+2)        // No of fcenter in North/South alg Vert codnate lines + 1 layr of Ghost cells
#define nyVfcGL  (nyVfcL+2)        // No of fcenter in East/West alg Vert codnate lines + 1 layr of Ghost cells
#define nVfcGL   (nxVfcGL*nyVfcGL)  // No of fcenter in Total alg Vert codnate lines + 1 layr of Ghost cells

#define dx (1.0/nxc)
#define dy (1.0/nyc)
#define dV (dx*dy)

