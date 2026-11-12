# Source this file to work in this worktree: it activates the development conda env and pins the GPU.
# Usage: source scripts/dev_env.sh   (then run Python and pytest from the worktree root)
#
# RADAR_DEV_CONDA_ENV   conda env to activate, default radar-rel (Channel 0.6.0, used read-only)
# RADAR_DEV_TOOLS_BIN   directory holding pinned ruff/mypy/clang-format; appended to PATH (it may hold its own python)
# CUDA_VISIBLE_DEVICES  set to 1 unless already set (an empty value, which hides CUDA, is kept)
# PYTHONPATH            this worktree first, so the CI gate scripts (run as `python ci/<gate>.py`) import its witwin.radar
# The native library is built separately with scripts/dev_build.sh.

_radar_dev_env() {
    local env=${RADAR_DEV_CONDA_ENV:-radar-rel} root
    root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
    if [[ ${CONDA_DEFAULT_ENV:-} != "$env" ]]; then
        local had_u=0
        [[ $- == *u* ]] && had_u=1
        set +u
        # shellcheck disable=SC1091
        source "${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh" && conda activate "$env"
        local status=$?
        [[ $had_u == 1 ]] && set -u
        [[ $status == 0 ]] || { echo "dev_env.sh: could not activate conda env $env" >&2; return 1; }
    fi
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-1}
    case ":${PYTHONPATH:-}:" in
        *":$root:"*) ;;
        *) export PYTHONPATH=$root${PYTHONPATH:+:$PYTHONPATH} ;;
    esac
    if [[ -n ${RADAR_DEV_TOOLS_BIN:-} ]]; then
        case ":$PATH:" in
            *":$RADAR_DEV_TOOLS_BIN:"*) ;;
            *) export PATH=$PATH:$RADAR_DEV_TOOLS_BIN ;;
        esac
    fi
    echo "dev_env.sh: conda env $env, PYTHONPATH starts at $root, CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
}
if ! _radar_dev_env; then
    unset -f _radar_dev_env
    return 1
fi
unset -f _radar_dev_env
