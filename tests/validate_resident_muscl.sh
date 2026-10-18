#!/bin/bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
mkdir -p "$root/runs"
check_dir=$(mktemp -d "$root/runs/resident_muscl_check_XXXXXXXX")
python3 - "$root" "$check_dir" <<'PYSETUP'
from pathlib import Path
import sys,shutil,re
root,base=map(Path,sys.argv[1:])
for steps in (3,100):
 for mode in ('off','on'):
  d=base/f'{steps}_{mode}'
  shutil.copytree(root/'input',d/'input')
  p=d/'input/ctrl.txt'; s=p.read_text()
  s=re.sub(r'^Final-Time\s+.*',f'Final-Time {steps*3e-8:.12g}',s,flags=re.M)
  for a in 'xyz': s=re.sub(r'^m_block_'+a+r'\s+.*',f'm_block_{a} 1',s,flags=re.M)
  p.write_text(s)
  p=d/'input/grid-3dRSBI.txt';p.write_text(re.sub(r'^total_nk\s+.*','total_nk 2',p.read_text(),flags=re.M))
PYSETUP
echo "Validation directory: $check_dir"
unset EU3D_FULL_RESIDENT EU3D_VALIDATE_FULL_RESIDENT EU3D_FAST_UNPACK EU3D_RESIDENT_MUSCL EU3D_VALIDATE_RESIDENT_MUSCL EU3D_TRACE_TEMP EU3D_VALIDATE_STATE EU3D_VALIDATE_CHAIN EU3D_VALIDATE_PACKED
export EU3D_CHAIN_STATE=1 EU3D_PACKED_OUTPUT=1 EU3D_EE_STATE=1 EU3D_REUSE_UPLOAD=1 EU3D_DUMP_STATE=1
for steps in 3 100; do
 for mode in off on; do
  (cd "$check_dir/${steps}_${mode}"
   export EU3D_RESIDENT_EE=1
   if [ "$mode" = on ]; then export EU3D_RESIDENT_MUSCL=1 EU3D_VALIDATE_RESIDENT_MUSCL=1 EU3D_VALIDATE_EE_STATE=1; else unset EU3D_RESIDENT_MUSCL EU3D_VALIDATE_RESIDENT_MUSCL EU3D_VALIDATE_EE_STATE; fi
   "$root/EU3D" > run.log 2>&1)
 done
 python3 "$root/tools/compare_states.py" "$check_dir/${steps}_off/output_1/results/state_check_1_rank0.bin" "$check_dir/${steps}_on/output_1/results/state_check_1_rank0.bin" > "$check_dir/compare${steps}.log"
 tail -n 1 "$check_dir/compare${steps}.log"
 test "$(grep -c 'Resident MUSCL validation: Passed' "$check_dir/${steps}_on/run.log")" -eq "$((steps-1))"
 test "$(grep -c 'EE state validation: Passed' "$check_dir/${steps}_on/run.log")" -eq "$steps"
 python3 - "$check_dir/compare${steps}.log" <<'PYCHECK'
import re,sys
s=open(sys.argv[1]).read()
values=re.findall(r'max_abs=([^, ]+)',s)
assert values and all(float(v)==0 for v in values), 'Expected exact field equality'
PYCHECK
done
