#!/bin/bash
set -euo pipefail
cd /root/private_data/liziye/EU3D-upload-reuse-20261007
project=$PWD
baseline=/root/private_data/liziye/EU3D-optimized-candidate
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
unset EU3D_TRACE_TEMP EU3D_VALIDATE_STATE EU3D_VALIDATE_CHAIN EU3D_VALIDATE_PACKED EU3D_VALIDATE_EE_STATE EU3D_REUSE_UPLOAD EU3D_DUMP_STATE
export EU3D_CHAIN_STATE=1 EU3D_PACKED_OUTPUT=1 EU3D_EE_STATE=1
for steps in 3 100; do
  if [ "$steps" = 3 ]; then final=9e-8; else final=3e-6; fi
  for mode in reference reuse1 reuse2; do
    dir="$project/check_${steps}_$mode"
    mkdir "$dir"
    cp -a input "$dir/"
    ranks=1; exe="$project/EU3D"; reuse=1
    if [ "$mode" = reference ]; then exe="$baseline/EU3D"; reuse=0; fi
    if [ "$mode" = reuse2 ]; then ranks=2; fi
    sed -i -e "s/^Final-Time[[:space:]].*/Final-Time $final/" -e "s/^m_block_x[[:space:]].*/m_block_x $ranks/" "$dir/input/ctrl.txt"
    (cd "$dir"; EU3D_DUMP_STATE=1 EU3D_REUSE_UPLOAD=$reuse mpirun --allow-run-as-root -np "$ranks" -x EU3D_DUMP_STATE -x EU3D_REUSE_UPLOAD -x EU3D_CHAIN_STATE -x EU3D_PACKED_OUTPUT -x EU3D_EE_STATE "$exe" > run.log 2>&1)
  done
  python3 compare_states.py "check_${steps}_reference/output_1/results/state_check_1_rank0.bin" "check_${steps}_reuse1/output_1/results/state_check_1_rank0.bin" > "compare_${steps}_single.log" 2>&1
  python3 compare_states.py "check_${steps}_reference/output_1/results/state_check_1_rank0.bin" "check_${steps}_reuse2/output_2/results/state_check_1_rank0.bin" "check_${steps}_reuse2/output_2/results/state_check_1_rank1.bin" > "compare_${steps}_dual.log" 2>&1
  tail -n 1 "compare_${steps}_single.log" "compare_${steps}_dual.log"
  # These changes only reuse identical data; require zero absolute differences.
  python3 - "$steps" <<'PY'
import re,sys
from pathlib import Path
for mode in ('single','dual'):
 s=Path(f'compare_{sys.argv[1]}_{mode}.log').read_text()
 values=re.findall(r'max_abs=([^, ]+)',s)
 assert values and all(float(v)==0 for v in values), 'Nonzero absolute error: '+mode
print('All compared fields have zero absolute difference.')
PY
done
