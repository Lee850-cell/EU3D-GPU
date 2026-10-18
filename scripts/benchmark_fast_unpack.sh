#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
mkdir -p "$root/runs"
bench=$(mktemp -d "$root/runs/fast_unpack_perf_XXXXXXXX")
python3 - "$root" "$bench" <<'PY'
from pathlib import Path
import sys,re,shutil
root,base=map(Path,sys.argv[1:])
for rep in range(1,4):
 for mode in ('off','on'):
  d=base/f'{mode}_{rep}';shutil.copytree(root/'input',d/'input')
  p=d/'input/ctrl.txt';s=re.sub(r'^Final-Time\s+.*','Final-Time 3e-6',p.read_text(),flags=re.M)
  for a in 'xyz':s=re.sub(r'^m_block_'+a+r'\s+.*',f'm_block_{a} 1',s,flags=re.M)
  p.write_text(s)
  p=d/'input/grid-3dRSBI.txt';p.write_text(re.sub(r'^total_nk\s+.*','total_nk 2',p.read_text(),flags=re.M))
PY
unset EU3D_FULL_RESIDENT EU3D_VALIDATE_FULL_RESIDENT EU3D_FAST_UNPACK EU3D_RESIDENT_MUSCL EU3D_VALIDATE_RESIDENT_MUSCL EU3D_VALIDATE_STATE EU3D_VALIDATE_CHAIN EU3D_VALIDATE_PACKED EU3D_VALIDATE_EE_STATE EU3D_TRACE_TEMP EU3D_DUMP_STATE
export EU3D_RESIDENT_MUSCL=1 EU3D_RESIDENT_EE=1 EU3D_CHAIN_STATE=1 EU3D_PACKED_OUTPUT=1 EU3D_EE_STATE=1 EU3D_REUSE_UPLOAD=1
echo "Benchmark directory: $bench"
for rep in 1 2 3; do
 for mode in off on; do
  (cd "$bench/${mode}_${rep}"
   if [ "$mode" = on ]; then export EU3D_FAST_UNPACK=1; else unset EU3D_FAST_UNPACK; fi
   "$root/EU3D" > run.log 2>&1)
 done
done
python3 - "$bench" <<'PY'
import re,statistics,sys,json
from pathlib import Path
b=Path(sys.argv[1]);out={}
for mode in ('off','on'):
 rows=[]
 for rep in range(1,4):
  s=(b/f'{mode}_{rep}'/'run.log').read_text()
  rows.append({k:float(re.search(r'^'+label+r': ([\d.]+)',s,re.M)[1]) for k,label in [('loop','Loop time'),('total','Total Time')]})
 out[mode]={'runs':rows,'median_loop':statistics.median(r['loop'] for r in rows),'median_total':statistics.median(r['total'] for r in rows)}
out['loop_reduction_percent']=100*(1-out['on']['median_loop']/out['off']['median_loop'])
(b/'summary.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
PY
