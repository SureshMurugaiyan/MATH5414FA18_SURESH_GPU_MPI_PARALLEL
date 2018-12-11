#	Name of the exexutable output
TARGET	= main
#VIRGINIA TECH#########################
# NVCC is path to nvcc. Here it is assumed that /usr/local/cuda is on one's PATH.
NVCC = nvcc

NVCCFLAGS = -arch=sm_35 -g -G
NVCCINC = -I$(CUDA_INC) -I$(VT_MPI_INC)
LFLAGS = -L$(VT_MPI_LIB) -lmpi -L$(CUDA_LIB64) 
##############################################

#
HDRDIR = ./include

#	Cpp Compiler:
CCpp	:= mpic++
#
#	Select a C++ dialec:
CppDial			= -std=c++11
#
#	Cpp Compiler Flags:
CppFLAGS		= -Wall
#
#	CUDA Compiler:
CCuda 		= nvcc


#
#	GPU architecture:
#GPUarch			= -arch=sm_61
#
#	Set CUDA Compiler Option		
#	you have to separate your different -Xcompiler sub-options with a comma 
#	or, you have to use for each option a separate -Xcompile.
SetSubOpt		= -Xcompiler
#
#	Cuda Linker Options:
LCuda			=	-lcuda -lcudart 
#
#	DEPENDENCIES:
CudaDEPS		= doNavier.cu\
	                  predictorCorrector.cu\
                          functions.cu\
                          finiteVolumeOperators.cu\
	                  poissonSolverPressure.cu\
				

CppDEPS			=input.h\
			 inputParallel.h\
			 inputSerial.h\

									


CppSourceFiles	= main.cpp\
                  mesh.cpp\
                  functionsParallel.cpp\
	          mpiCheck.cpp\
                  parallelSubroutines.cpp\
	          printData.cpp\
	          solveParallel.cpp\
	          solveSerial.cpp\
	          timeStep.cpp\

CPPobjects	= main.o\
                  mesh.o\
                  functionsParallel.o\
	          mpiCheck.o\
                  parallelSubroutines.o\
	          printData.o\
	          solveParallel.o\
	          solveSerial.o\
	          timeStep.o\
			

CUDAobjects 	= doNavier.o\
	          predictorCorrector.o\
                  functions.o\
                  finiteVolumeOperators.o\
	          poissonSolverPressure.o\
########################################################################

all: $(CPPobjects) $(CUDAobjects)
	# MAKING AND LINKING SUCCESS
#	$(NVCC) $(GPUarch) -ccbin $(CCpp) -Xcompiler $(CppDial) $(CUDAobjects)  $(CppSourceFiles)  $(LCuda) -I$(CUDA_INC) -I$(OMPI_INC) -L$(OMPI_LIB) -lmpi -L$(CUDA_LIB64) -o $(TARGET)
	$(NVCC) $(NVCCFLAGS) $(NVCCINC) -ccbin $(CCpp) -Xcompiler $(CppDial) $(CUDAobjects)  $(CppSourceFiles)  $(LCuda)  -o $(TARGET)
		


%.o: %.cpp $(CppDEPS)
	# Compiling the Cpp files: 
	$(CCpp) $(CppDial) -c -o $@ $< $(CppFLAGS)
	


%.o: %.cu
	# Compiling the Kernel files: 
	#$(CCuda) -c $(GPUarch) $(CppDial) CudaOperations.cu
	$(NVCC) -c $(NVCCFLAGS) $(NVCCINC) $(LFLAGS) $(CppDial) $(CudaDEPS)


.PHONY: clean
clean:
		rm -f *.o *.gch $(TARGET)
########################################################################
#
#
#  	nvcc -c -arch=sm_61 CudaKernels.cu
#  	nvcc -ccbin g++ -Xcompiler "-std=c++11" CudaKernels.o  main.cpp ...
#	lodepng.cpp CudaHelper.cpp  -lcuda -lcudart  -o julia
#
#	-c 		flag says to generate the object file, 
#	-o $@ 	says to put the output of the compilation in the file named 
#			on the left side of the ':'.
# 	$< 		is the first item in the dependencies list. 

