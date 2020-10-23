#!/bin/bash

#PBS -P z67 
#PBS -q normal
#PBS -l ncpus=528
#PBS -l mem=528GB
#PBS -l walltime=10:00:00
#PBS -l storage=gdata/qi71+gdata/z67
#PBS -l wd
#PBS -N galeirjmcmctdem
#PBS -o galeirjmcmctdem.out
#PBS -e galeirjmcmctdem.err
#PBS -j oe
module load openmpi/4.0.1
module load fftw3/3.3.8
module load petsc/3.12.2
module load netcdf/4.7.1
module load gdal/3.0.2
export PATH=$HOME/ga-aem/bin/gadi/intel:$PATH
which garjmcmctdem.exe
mpirun garjmcmctdem.exe garjmcmctdem-vanilla.con
