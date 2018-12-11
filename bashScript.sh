#! /bin/bash
#
#PBS -l walltime=00:10:00
#PBS -l nodes=4:ppn=1:gpus=1
#PBS -W group_list=newriver
#PBS -q p100_normal_q
#PBS -A SPHysics
#PBS -M sureshm@vt.edu
#
cd $PBS_O_WORKDIR
#
module purge
module load gcc
module load openmpi
module load cuda
#
# compile
#
make
#
 mpiexec -npernode 1 ./main
#
#nvprof --metrics all ./main

# clean up
#
#rm vecadd
#
echo "Normal end of script."
