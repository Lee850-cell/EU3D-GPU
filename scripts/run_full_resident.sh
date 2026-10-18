#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export EU3D_RESIDENT_EE=1 EU3D_RESIDENT_MUSCL=1 EU3D_FULL_RESIDENT=1
unset EU3D_FAST_UNPACK
exec bash "$root/scripts/run_local.sh" "${1:-3}"
