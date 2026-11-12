#!/usr/bin/env bash
# Build this worktree's native prebuilt (witwin/radar/cuda/prebuilt/) for local development on Linux.
# Usage: scripts/dev_build.sh [--verbose]   (arguments go to scripts/build_radar_cuda_prebuilt.py)
#
# RADAR_DEV_CONDA_ENV   conda env used read-only, default radar-rel
# RADAR_DEV_CUDA_ARCH   TORCH_CUDA_ARCH_LIST entry, default 8.6
# The build directory is the loader's source-keyed default under the temp dir. A killed build leaves a
# torch FileBaton 'lock' there with no library beside it, and the next build would wait on it forever,
# so the script refuses to start on one. A lock file in build/ serializes builds of this worktree.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
set +u
# shellcheck disable=SC1091
source "${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate "${RADAR_DEV_CONDA_ENV:-radar-rel}"
set -u
E=$CONDA_PREFIX
# Conda's CUDA: the real nvcc (the targets/ symlink has no nvcc.profile) and the env as CUDA_HOME.
export CUDA_HOME=$E CUDA_NVCC_EXECUTABLE=$E/bin/nvcc
export TORCH_CUDA_ARCH_LIST=${RADAR_DEV_CUDA_ARCH:-8.6}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
unset WITWIN_CUDA_GENCODE_ARCHES

mkdir -p "$ROOT/build"
exec 9>"$ROOT/build/.dev_build.lock"
if ! flock -n 9; then
    echo "dev_build.sh: another build of this worktree is running; waiting for it" >&2
    flock 9
fi

cd "$ROOT"
BUILD_DIR=${WITWIN_RADAR_NATIVE_BUILD_DIR:-$(python -c 'import runpy; print(runpy.run_path("witwin/radar/cuda/runtime.py")["default_build_directory"]())')}
while IFS= read -r lock; do
    if ! compgen -G "$(dirname "$lock")/*.so" >/dev/null; then
        echo "dev_build.sh: stale build lock without a library: $lock (a killed build; remove it and rerun)" >&2
        exit 1
    fi
done < <(find "$BUILD_DIR" -name lock -type f 2>/dev/null)

python scripts/build_radar_cuda_prebuilt.py "$@"
