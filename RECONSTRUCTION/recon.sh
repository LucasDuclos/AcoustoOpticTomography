#! /bin/bash

# Reconstruction script for Acousto-Optic Tomography
# Model : Y = A*L
# Make the reconstruction of L -> L_recon

Y_path="pathTO/AOSignals.cdh"
A_path="pathTO/system_matrix/"
L_recon_Path="PathTo/results/"

CASToR_path="/home/duclos/AOT/Reconstruction/CASToR_LuK/bin/castor-recon"

iter="100:10"
opti="MLEM" #MLEM DEPIERRO95
penalty="" # "-pnlt MRF:MRF.conf -pnlt-beta 0.4"


if [ ! -f ${Y_path} ]; then
  echo "$0> Error: no input file ${Y_path}"
  exit 1
elif [ ! -d ${A_path} ]; then
  echo "$0> Error: no system matrix directory ${A_path}"
  exit 2
fi  


cmd="${CASToR_path} -df ${Y_path} -opti ${opti} ${penalty} -it ${iter}"
cmd="${cmd} -proj matrix -dout ${imageDir} -th 24 -vb 5 -proj-comp 1 -ignore-scanner"
cmd="${cmd} -data-type AOT -ignore-corr cali,fdur"
cmd="${cmd} -system-matrix ${A_path}"
echo ${cmd}
${cmd}
#castor-recon -df AOSignals.cdh -opti LDWB -it 100:1 -proj matrix -dout results -th 1 -vb 5 -proj-comp 1 -ignore-scanner -data-type AOT -ignore-corr cali,fdur -system-matrix ../../system_matrix/
exit 0
