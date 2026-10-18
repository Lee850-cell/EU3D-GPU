#!/usr/bin/env bash
# Run a small single-GPU case in its own directory; keep input templates intact.
set -euo pipefail
project=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
steps=${1:-3}
case "$steps" in
  3) final=9e-8 ;;
  100) final=3e-6 ;;
  *) echo 'Usage: bash scripts/run_local.sh [3|100]' >&2; exit 2 ;;
esac
test -x "$project/EU3D" || { echo 'Compile first: make -j2' >&2; exit 1; }
mkdir -p "$project/runs"
run=$(mktemp -d "$project/runs/local_${steps}_XXXXXXXX")
cp -a "$project/input" "$run/input"
sed -i -e "s/^Final-Time[[:space:]].*/Final-Time $final/" \
  -e 's/^m_block_x[[:space:]].*/m_block_x 1/' \
  -e 's/^m_block_y[[:space:]].*/m_block_y 1/' \
  -e 's/^m_block_z[[:space:]].*/m_block_z 1/' "$run/input/ctrl.txt"
sed -i 's/^total_nk[[:space:]].*/total_nk 2/' "$run/input/grid-3dRSBI.txt"
unset EU3D_VALIDATE_FULL_RESIDENT EU3D_VALIDATE_RESIDENT_MUSCL EU3D_VALIDATE_STATE EU3D_VALIDATE_CHAIN EU3D_VALIDATE_PACKED EU3D_VALIDATE_EE_STATE EU3D_TRACE_TEMP EU3D_DUMP_STATE
export EU3D_CHAIN_STATE=1 EU3D_PACKED_OUTPUT=1 EU3D_EE_STATE=1 EU3D_REUSE_UPLOAD=1
echo "Run directory: $run"
cd "$run"
"$project/EU3D" > run.log 2>&1
grep -E 'Loop time:|Total Time:' run.log
