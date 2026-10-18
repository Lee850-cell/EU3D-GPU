#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
mkdir -p "$root/runs"
check_dir=$(mktemp -d "$root/runs/full_resident_check_XXXXXXXX")
case "${1:-thin}" in
 thin) ni=281; nj=141; nk=2 ;;
 volume) ni=17; nj=13; nk=4 ;;
 *) echo 'Usage: validate_full_resident.sh [thin|volume]' >&2; exit 2 ;;
esac
python3 - "$root" "$check_dir" "$ni" "$nj" "$nk" <<'PY'
from pathlib import Path
import sys,shutil,re
root,base=map(Path,sys.argv[1:3])
ni,nj,nk=map(int,sys.argv[3:6])
for steps in (3,100):
 for mode in ('off','shadow','on'):
  d=base/f'{steps}_{mode}';shutil.copytree(root/'input',d/'input')
  p=d/'input/ctrl.txt';s=re.sub(r'^Final-Time\s+.*',f'Final-Time {steps*3e-8:.12g}',p.read_text(),flags=re.M)
  for a in 'xyz': s=re.sub(r'^m_block_'+a+r'\s+.*',f'm_block_{a} 1',s,flags=re.M)
  # Exercise intermediate output/synchronization as well as final state dump.
  s=re.sub(r'^count\s+.*','count 2',s,flags=re.M)
  p.write_text(s)
  p=d/'input/grid-3dRSBI.txt';s=p.read_text()
  for axis,n in [('i',ni),('j',nj),('k',nk)]:s=re.sub(r'^total_n'+axis+r'\s+.*',f'total_n{axis} {n}',s,flags=re.M)
  p.write_text(s)
PY
echo "Validation directory: $check_dir"
unset EU3D_FAST_UNPACK EU3D_FULL_RESIDENT EU3D_VALIDATE_FULL_RESIDENT EU3D_VALIDATE_RESIDENT_MUSCL EU3D_VALIDATE_EE_STATE EU3D_TRACE_TEMP EU3D_VALIDATE_STATE EU3D_VALIDATE_CHAIN EU3D_VALIDATE_PACKED
export EU3D_CHAIN_STATE=1 EU3D_PACKED_OUTPUT=1 EU3D_EE_STATE=1 EU3D_REUSE_UPLOAD=1 EU3D_DUMP_STATE=1 EU3D_RESIDENT_EE=1 EU3D_RESIDENT_MUSCL=1
for steps in 3 100; do
 for mode in off shadow on; do
  (cd "$check_dir/${steps}_${mode}"
   if [ "$mode" != off ]; then export EU3D_FULL_RESIDENT=1; fi
   if [ "$mode" = shadow ]; then export EU3D_VALIDATE_FULL_RESIDENT=1; fi
   "$root/EU3D" > run.log 2>&1)
 done
 for mode in shadow on; do
  python3 "$root/tools/compare_states.py" "$check_dir/${steps}_off/output_1/results/state_check_2_rank0.bin" "$check_dir/${steps}_${mode}/output_1/results/state_check_2_rank0.bin" > "$check_dir/compare${steps}_${mode}.log"
  tail -n 1 "$check_dir/compare${steps}_${mode}.log"
 done
 python3 "$root/tools/compare_states.py" "$check_dir/${steps}_shadow/output_1/results/state_check_2_rank0.bin" "$check_dir/${steps}_on/output_1/results/state_check_2_rank0.bin" > "$check_dir/shadow_vs_on${steps}.log"
 python3 - "$check_dir/shadow_vs_on${steps}.log" <<'PYCHECK'
import re,sys
values=re.findall(r'max_abs=([^, ]+)',open(sys.argv[1]).read())
assert values and all(float(v)==0 for v in values), 'Deferred synchronization must preserve exact state'
PYCHECK
 for output in 1 2; do
  cmp "$check_dir/${steps}_shadow/output_1/results/SeriesTrapezoid-Adaptive${output}_0.dat" "$check_dir/${steps}_on/output_1/results/SeriesTrapezoid-Adaptive${output}_0.dat"
 done
 cmp "$check_dir/${steps}_off/output_1/load/Load_Loop.dat" "$check_dir/${steps}_on/output_1/load/Load_Loop.dat"
 test "$(grep -c 'Resident health: invalid_nchem=0' "$check_dir/${steps}_on/run.log")" -eq "$steps"
 test "$(grep -c 'Full resident: deferred' "$check_dir/${steps}_on/run.log")" -eq "$steps"
 test "$(grep -c 'Resident output sync:' "$check_dir/${steps}_on/run.log")" -eq 2
 test "$(grep -c 'Resident MUSCL validation: Passed' "$check_dir/${steps}_shadow/run.log")" -eq "$((steps-1))"
done
