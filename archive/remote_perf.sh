#!/bin/bash
set -euo pipefail
cd /root/private_data/liziye/EU3D-upload-reuse-20261007
project=$PWD
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
unset EU3D_TRACE_TEMP EU3D_VALIDATE_STATE EU3D_VALIDATE_CHAIN EU3D_VALIDATE_PACKED EU3D_VALIDATE_EE_STATE EU3D_DUMP_STATE
export EU3D_CHAIN_STATE=1 EU3D_PACKED_OUTPUT=1 EU3D_EE_STATE=1
for trial in 1 2 3; do
  for reuse in 0 1; do
    dir="$project/perf_reuse${reuse}_${trial}"
    mkdir "$dir"; cp -a input "$dir/"
    sed -i -e 's/^Final-Time[[:space:]].*/Final-Time 3e-6/' -e 's/^m_block_x[[:space:]].*/m_block_x 2/' "$dir/input/ctrl.txt"
    (cd "$dir"; EU3D_REUSE_UPLOAD=$reuse mpirun --allow-run-as-root -np 2 -x EU3D_REUSE_UPLOAD -x EU3D_CHAIN_STATE -x EU3D_PACKED_OUTPUT -x EU3D_EE_STATE "$project/EU3D" > run.log 2>&1)
    grep -H -E 'Loop time:|Total Time:' "$dir/run.log"
  done
done
