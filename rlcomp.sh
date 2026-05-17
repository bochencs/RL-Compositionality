#!/usr/bin/env bash
# RL-Compositionality — single-entrypoint script for offline environment setup,
# training, inference, and shared-filesystem logs.
#
# This repo carries a file-based offline environment contract:
#   - bootstrap-cache  populates .python/ and wheels/ on a networked machine.
#   - fetch-assets     populates model_store/ and data/string_task/.
#   - install          rebuilds .venv-rlcomp/ without network access.
#   - doctor           verifies the local offline environment and smoke imports.
#
# USAGE
# -----
# Source mode (set env in the CURRENT shell; no action run):
#     source /path/to/RL-Compositionality/rlcomp.sh
#
# Execute mode (sets env in a subshell, runs ACTION, then exits):
#     bash /path/to/RL-Compositionality/rlcomp.sh ACTION [options]
#
# Actions:
#     env       Source bootstrap.sh and run a sanity check.
#     prepare   Generate the newops datasets.
#     stage1    Stage 1 RFT (vLLM rollout -> filter -> FSDP SFT).
#     stage2    Stage 2 GRPO (needs >=4 GPU for full 8B model).
#     infer     Inference matrix.
#     all       prepare -> stage1 -> stage2 -> infer (serial).
#     install   Rebuild .venv-rlcomp/ from repo-local Python + wheels.
#     doctor    Diagnose offline env, imports, CUDA, assets, and paths.
#     bootstrap-cache  Populate .python/ and wheels/ on a networked machine.
#     fetch-assets     Download HF model/data assets into repo-local paths.
#     package-offline  Verify or tar the complete offline bundle.
#     logs      View pipeline logs. Sub: (none) | tail | list | cat | show <id>
#
# Every invocation writes a log dir under
# results/pipeline_runs/<ts>_<host>_<action>/ with terminal stdout/stderr,
# per-action logs, metadata, history.jsonl, and a `latest` symlink.
# Use `rlcomp.sh logs` to browse those runs from any machine that sees the
# shared repository filesystem.

_RLCOMP_SELF="${BASH_SOURCE[0]:-$0}"
_RLCOMP_HERE="$(cd "$(dirname "${_RLCOMP_SELF}")" && pwd)"
_RLCOMP_BOOT="${_RLCOMP_HERE}/scripts/env/bootstrap.sh"
_RLCOMP_RUN="${_RLCOMP_HERE}/scripts/env/run.sh"
_RLCOMP_LOGS="${_RLCOMP_HERE}/scripts/env/logs.sh"
_RLCOMP_LIB_LOGGING="${_RLCOMP_HERE}/scripts/env/lib_logging.sh"
_RLCOMP_INSTALL="${_RLCOMP_HERE}/scripts/env/install_hermetic.sh"
_RLCOMP_DOCTOR="${_RLCOMP_HERE}/scripts/env/doctor.sh"
_RLCOMP_FREEZE="${_RLCOMP_HERE}/scripts/env/freeze.sh"
_RLCOMP_CACHE_BOOTSTRAP="${_RLCOMP_HERE}/scripts/env/bootstrap_offline_cache.sh"
_RLCOMP_FETCH_ASSETS="${_RLCOMP_HERE}/scripts/env/fetch_assets.sh"
_RLCOMP_PACKAGE_OFFLINE="${_RLCOMP_HERE}/scripts/env/package_offline_bundle.sh"

if [ ! -f "${_RLCOMP_BOOT}" ]; then
    echo "[rlcomp] ERROR: cannot find ${_RLCOMP_BOOT}" >&2
    echo "[rlcomp]        Is this the RL-Compositionality repo root?" >&2
    return 1 2>/dev/null || exit 1
fi

# ---------- source mode: set env and return, leaving caller's shell intact ----------
# (Do NOT apply `set -e` / `set -u` here — it would leak into the caller.)
if [ "${BASH_SOURCE[0]:-}" != "${0}" ]; then
    # shellcheck disable=SC1090
    source "${_RLCOMP_BOOT}"
    return 0 2>/dev/null || true
fi

# ---------- execute mode: strict shell, single ACTION ----------
set -euo pipefail
ACTION="${1:-}"
if [ -z "${ACTION}" ]; then
    sed -n '2,40p' "${_RLCOMP_SELF}"
    exit 2
fi
shift || true

# ---------- outer terminal mirror ----------
# Any node on the shared filesystem that invokes `bash rlcomp.sh <ACTION>`
# gets a self-contained capture of everything the terminal sees, from this
# point until the shell exits. This includes the bootstrap banner, run.sh's
# per-sub-action output, pipeline summaries, and anything printed by a
# subprocess. Files land in:
#
#   results/pipeline_runs/<ts>_<host>_<action>/terminal.stdout.log
#   results/pipeline_runs/<ts>_<host>_<action>/terminal.stderr.log
#
# We set this up BEFORE sourcing bootstrap.sh or dispatching, so the capture
# starts from the very first line that dispatch prints. The inner loggers in
# run.sh / lib_logging.sh detect RLCOMP_PIPELINE_RUN_DIR and reuse the same
# directory rather than creating a nested one.
#
# Skipped for viewer / help / unknown actions — they exit fast with usage text
# and produce nothing worth preserving.
case "${ACTION}" in
    env|prepare|stage1|stage2|infer|all|install|doctor|freeze|bootstrap-cache|cache|fetch-assets|assets|package-offline|bundle)
        _rl_ts="$(date +%Y%m%d_%H%M%S)"
        _rl_host="$(hostname 2>/dev/null | tr -d '\r\n' | tr -c 'A-Za-z0-9_-' '-' | cut -c1-32)"
        _rl_host="${_rl_host:-unknown}"
        _rl_runs_root="${_RLCOMP_HERE}/results/pipeline_runs"
        if ! mkdir -p "${_rl_runs_root}" 2>/dev/null || [ ! -w "${_rl_runs_root}" ]; then
            _rl_runs_root="${_RLCOMP_HERE}/logs/pipeline_runs"
            mkdir -p "${_rl_runs_root}" 2>/dev/null || _rl_runs_root=""
            [ -w "${_rl_runs_root}" ] || _rl_runs_root=""
        fi
        _rl_run_dir=""
        if [ -n "${_rl_runs_root}" ]; then
            _rl_run_dir="${_rl_runs_root}/${_rl_ts}_${_rl_host}_${ACTION}"
            _rl_dup=1
            while [ -e "${_rl_run_dir}" ]; do
                _rl_run_dir="${_rl_runs_root}/${_rl_ts}_${_rl_host}_${ACTION}_${_rl_dup}"
                _rl_dup=$((_rl_dup + 1))
            done
            if ! mkdir -p "${_rl_run_dir}" 2>/dev/null; then
                echo "[rlcomp] WARN: could not create ${_rl_run_dir}; terminal mirror disabled" >&2
                _rl_run_dir=""
            fi
        fi
        if [ -n "${_rl_run_dir}" ]; then
            export RLCOMP_PIPELINE_RUN_DIR="${_rl_run_dir}"
            export RLCOMP_PIPELINE_TS="${_rl_ts}"
            export RLCOMP_PIPELINE_HOST="${_rl_host}"
            export RLCOMP_PIPELINE_ACTION="${ACTION}"
            # Thin header so readers can identify the dir even if the inner
            # logger (run.sh / lib_logging.sh) never writes its own meta.
            {
                printf 'action: %s\n' "${ACTION}"
                printf 'ts: %s\n' "${_rl_ts}"
                printf 'host: %s\n' "${_rl_host}"
                printf 'pid: %s\n' "$$"
                printf 'argv:%s\n' "$(printf ' %q' "${ACTION}" "$@")"
                printf 'pwd: %s\n' "$(pwd 2>/dev/null || echo unknown)"
                printf 'user: %s@%s\n' "${USER:-unknown}" "$(hostname 2>/dev/null || echo unknown)"
                printf 'repo_root: %s\n' "${_RLCOMP_HERE}"
                printf 'run_dir: %s\n' "${_rl_run_dir}"
                printf 'started_utc: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
            } > "${_rl_run_dir}/wrapper.context.txt" 2>/dev/null || true
            # Outer tee: stream this shell's stdout/stderr to BOTH the terminal
            # and log files from now until exit. Subsequent `exec bash run.sh`
            # or function calls inherit these FDs, so nothing escapes capture.
            exec \
                > >(tee -a "${_rl_run_dir}/terminal.stdout.log") \
                2> >(tee -a "${_rl_run_dir}/terminal.stderr.log" >&2)
            printf '[rlcomp] terminal mirror active\n' >&2
            printf '[rlcomp]   run_dir: %s\n' "${_rl_run_dir}" >&2
            printf '[rlcomp]   host:    %s\n' "${_rl_host}" >&2
            printf '[rlcomp]   action:  %s\n' "${ACTION}" >&2
        fi
        unset _rl_ts _rl_host _rl_runs_root _rl_run_dir _rl_dup
        ;;
esac

_rlcomp_wrap() {
    local action="$1"
    shift
    if [ -f "${_RLCOMP_LIB_LOGGING}" ]; then
        # shellcheck disable=SC1090
        source "${_RLCOMP_LIB_LOGGING}"
        rlcomp_log_wrap "${action}" "$@"
    else
        echo "[rlcomp] WARN: logging helper missing; running without inner logs" >&2
        "$@"
    fi
}

case "${ACTION}" in
    env|prepare|stage1|stage2|infer|all)
        # run.sh already owns its own per-invocation logging. Do NOT wrap.
        exec bash "${_RLCOMP_RUN}" "${ACTION}" "$@"
        ;;
    install)
        _rlcomp_wrap install bash "${_RLCOMP_INSTALL}" "$@"
        ;;
    doctor)
        _rlcomp_wrap doctor bash "${_RLCOMP_DOCTOR}" "$@"
        ;;
    freeze)
        _rlcomp_wrap freeze bash "${_RLCOMP_FREEZE}" "$@"
        ;;
    bootstrap-cache|cache)
        _rlcomp_wrap bootstrap-cache bash "${_RLCOMP_CACHE_BOOTSTRAP}" "$@"
        ;;
    fetch-assets|assets)
        _rlcomp_wrap fetch-assets bash "${_RLCOMP_FETCH_ASSETS}" "$@"
        ;;
    package-offline|bundle)
        _rlcomp_wrap package-offline bash "${_RLCOMP_PACKAGE_OFFLINE}" "$@"
        ;;
    logs)
        exec bash "${_RLCOMP_LOGS}" "$@"
        ;;
    -h|--help)
        sed -n '2,40p' "${_RLCOMP_SELF}"
        ;;
    *)
        echo "[rlcomp] Unknown action: ${ACTION}" >&2
        echo "[rlcomp] Try: env | prepare | stage1 | stage2 | infer | all | install | doctor | bootstrap-cache | fetch-assets | package-offline | logs" >&2
        exit 2
        ;;
esac
