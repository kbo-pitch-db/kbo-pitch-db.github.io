#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
workspace="$(cd -- "$script_dir/../.." && pwd)"
mkdir -p "$workspace/dataset"
exec > >(tee -a "$workspace/dataset/wham_run.log") 2>&1
exec python3 "$script_dir/setup_wham.py" "$@"
