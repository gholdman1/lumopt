#!/bin/sh

nthreads=$1
fsp_file=$2

module load lumerical
/opt/lumerical/fdtd/mpich2/nemesis/bin/mpiexec -n ${nthreads} /opt/lumerical/fdtd/bin/fdtd-engine-mpich2nem ${fsp_file}

cp ${fsp_file} run_${fsp_file}
