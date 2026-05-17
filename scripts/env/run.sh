#!/usr/bin/env bash
# RL-Compositionality unified entry point.
#
# Can be invoked from ANY cwd (including a freshly spawned shell). It first
# sources bootstrap.sh to set up env, then dispatches to the appropriate
# experiment script with the right env vars forwarded.
#
# Usage:
#     bash scripts/env/run.sh <action> [options...]     (from repo root)
#     bash /path/to/RL-Compositionality/scripts/env/run.sh <action> [options...]
#
#     actions:
#       prepare           Generate all newops parquet datasets.
#       stage1            Run Stage 1 (RFT/SFT).
#       stage2            Run Stage 2 (GRPO RL).
#       infer             Run the newops inference matrix on the current stage1 model.
#       all               prepare -> stage1 -> stage2 -> infer
#       env               Just source bootstrap and print env (sanity check).
#
#     common options (env passthrough):
#       CASE=baseline|newops          [default: newops]
#       AUTO_RESUME=0|1               [default: 1]
#       N_GPUS_PER_NODE=N             [default: auto-detect]
#       SAVE_FREQ=N                   [default: 50]
#       REMOVE_PREVIOUS_CKPT_IN_SAVE=True|False  [default: True]
#       WANDB_MODE=offline|online     [default: online if WANDB_API_KEY set]
#       RLCOMP_DRY_RUN=1              Print commands but do not execute.
#
# Logging:
#     Each invocation creates  results/pipeline_runs/<ts>_<action>/  containing
#     per-step stdout.log, stderr.log, meta.json. Top-level invocation.meta.json
#     and an entry in results/pipeline_runs/history.jsonl are written on exit.
#     results/pipeline_runs/latest always points to the most recent invocation.

set -euo pipefail

_RUN_SRC="${BASH_SOURCE[0]}"
_RUN_DIR="$(cd "$(dirname "${_RUN_SRC}")" && pwd)"

# shellcheck disable=SC1091
source "${_RUN_DIR}/bootstrap.sh"

cd "${RLCOMP_REPO_ROOT}"

ACTION="${1:-}"
shift || true

if [ -z "${ACTION}" ]; then
    echo "Usage: bash scripts/env/run.sh <env|prepare|stage1|stage2|infer|all>" >&2
    exit 2
fi

DRY_RUN="${RLCOMP_DRY_RUN:-0}"

# ============================================================================
# pipeline logging system
# ============================================================================

_pipeline_init_logdir() {
    # If rlcomp.sh (parent wrapper) already built a run dir for this invocation,
    # reuse it so every piece of output — the outer terminal mirror AND the
    # per-sub-action stdout/stderr logs — lands in the same directory. Without
    # this, every call would nest a fresh timestamped dir inside the parent's,
    # splitting the logs into two trees that are hard to correlate.
    if [ -n "${RLCOMP_PIPELINE_RUN_DIR:-}" ] && [ -d "${RLCOMP_PIPELINE_RUN_DIR}" ]; then
        RLCOMP_PIPELINE_TS="${RLCOMP_PIPELINE_TS:-$(date +%Y%m%d_%H%M%S)}"
        RLCOMP_PIPELINE_ACTION="${ACTION}"
    else
        RLCOMP_PIPELINE_TS="$(date +%Y%m%d_%H%M%S)"
        RLCOMP_PIPELINE_ACTION="${ACTION}"
        RLCOMP_PIPELINE_RUN_DIR="${RLCOMP_REPO_ROOT}/results/pipeline_runs/${RLCOMP_PIPELINE_TS}_${ACTION}"
        mkdir -p "${RLCOMP_PIPELINE_RUN_DIR}"
    fi
    RLCOMP_PIPELINE_START_ISO="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    RLCOMP_PIPELINE_START_EPOCH="$(date +%s)"
    export RLCOMP_PIPELINE_TS RLCOMP_PIPELINE_ACTION RLCOMP_PIPELINE_RUN_DIR \
           RLCOMP_PIPELINE_START_ISO RLCOMP_PIPELINE_START_EPOCH
    # Capture the bootstrap banner and initial env for forensics.
    {
        echo "=== invocation ==="
        echo "action: ${ACTION}"
        echo "ts: ${RLCOMP_PIPELINE_TS}"
        echo "run_dir: ${RLCOMP_PIPELINE_RUN_DIR}"
        echo "argv: $(printf ' %q' "$@")"
        echo "pwd: $(pwd)"
        echo "user: $(whoami)@$(hostname)"
        echo ""
        echo "=== bootstrap env (filtered) ==="
        env \
            | grep -E "^(RLCOMP_|GPU_|MODEL_|ACTOR_|REF_|CPU_|OFFLOAD_|GRAD_|SAVE_|TEST_|N_GPUS|NNODES|NPROC|HF_|TMPDIR|RAY_|WANDB_MODE|WANDB_PROJECT|WANDB_DIR|VIRTUAL_ENV|PYTHONPATH)=" \
            | grep -Ev "(API_KEY|TOKEN|SECRET|PASSWORD)=" \
            | sort
    } > "${RLCOMP_PIPELINE_RUN_DIR}/invocation.context.txt" 2>/dev/null || true
    trap _pipeline_finalize EXIT
}

_pipeline_finalize() {
    local exit_code=$?
    local end_iso end_epoch duration
    end_iso="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    end_epoch="$(date +%s)"
    duration=$((end_epoch - RLCOMP_PIPELINE_START_EPOCH))
    # Defer to a python helper for JSON/symlink/history bookkeeping.
    "${RLCOMP_PYTHON}" - <<'PY' "${RLCOMP_PIPELINE_RUN_DIR}" "${RLCOMP_PIPELINE_ACTION}" \
        "${RLCOMP_PIPELINE_START_ISO}" "${end_iso}" "${duration}" "${exit_code}" \
        "${RLCOMP_REPO_ROOT}" 2>/dev/null || true
import json, os, subprocess, sys

run_dir, action, start_iso, end_iso, duration, exit_code, repo_root = sys.argv[1:8]
pipeline_runs_dir = os.path.dirname(run_dir)
run_id = os.path.basename(run_dir)

# git commit
git_commit = "unknown"
try:
    git_commit = subprocess.check_output(
        ["git", "-C", repo_root, "rev-parse", "HEAD"],
        stderr=subprocess.DEVNULL, text=True
    ).strip()
except Exception:
    pass

# gpu info
gpu_info = "none"
try:
    gpu_info = subprocess.check_output(
        ["nvidia-smi", "-L"], stderr=subprocess.DEVNULL, text=True
    ).strip()
except Exception:
    pass

# Collect sub-action metas (one per step in do_all, or one for single action)
sub_actions = []
for f in sorted(os.listdir(run_dir)):
    if f.endswith(".meta.json") and f != "invocation.meta.json":
        try:
            with open(os.path.join(run_dir, f)) as fh:
                sub_actions.append(json.load(fh))
        except Exception:
            pass

# Filter env vars for snapshot. Drop anything that looks like a secret.
SECRET_SUBSTRINGS = ("API_KEY", "TOKEN", "SECRET", "PASSWORD")
ENV_PREFIXES = ("RLCOMP_", "GPU_", "MODEL_", "ACTOR_", "REF_", "CPU_",
                "OFFLOAD_", "GRAD_", "SAVE_", "TEST_", "N_GPUS", "NNODES",
                "NPROC", "HF_", "TMPDIR", "RAY_", "WANDB_MODE",
                "WANDB_PROJECT", "WANDB_DIR", "VIRTUAL_ENV", "PYTHONPATH")
env_snapshot = {
    k: v for k, v in os.environ.items()
    if k.startswith(ENV_PREFIXES)
    and not any(s in k.upper() for s in SECRET_SUBSTRINGS)
}

invocation = {
    "run_id": run_id,
    "action": action,
    "start_time_utc": start_iso,
    "end_time_utc": end_iso,
    "duration_sec": int(duration),
    "exit_code": int(exit_code),
    "host": os.uname().nodename,
    "git_commit": git_commit,
    "n_gpus_per_node": os.environ.get("N_GPUS_PER_NODE", "?"),
    "gpu_info": gpu_info,
    "env_snapshot": env_snapshot,
    "sub_actions": sub_actions,
}
with open(os.path.join(run_dir, "invocation.meta.json"), "w") as f:
    json.dump(invocation, f, indent=2)

# history.jsonl — flat, one line per invocation, no env snapshot (keep grep-able)
hist_entry = {k: v for k, v in invocation.items() if k != "env_snapshot"}
hist_entry["sub_actions"] = [
    {"sub_action": s.get("sub_action"), "exit_code": s.get("exit_code"),
     "duration_sec": s.get("duration_sec")}
    for s in sub_actions
]
os.makedirs(pipeline_runs_dir, exist_ok=True)
with open(os.path.join(pipeline_runs_dir, "history.jsonl"), "a") as f:
    f.write(json.dumps(hist_entry) + "\n")

# labbook.md — human-readable
lb = os.path.join(pipeline_runs_dir, "labbook.md")
first = not os.path.exists(lb)
with open(lb, "a") as f:
    if first:
        f.write("# RL-Compositionality pipeline runs\n\n"
                "Auto-generated by `scripts/env/run.sh`. One section per invocation.\n\n")
    status = "OK" if int(exit_code) == 0 else f"FAIL ({exit_code})"
    f.write(f"## `{run_id}` — {status}, {int(duration)}s\n\n")
    f.write(f"- host: `{invocation['host']}`\n")
    f.write(f"- git: `{git_commit[:10]}`\n")
    f.write(f"- gpus: {invocation['n_gpus_per_node']}\n")
    if sub_actions:
        f.write(f"- sub-actions:\n")
        for s in sub_actions:
            name = s.get("sub_action", "?")
            ex = s.get("exit_code", "?")
            dur = s.get("duration_sec", "?")
            f.write(f"  - `{name}`: exit {ex}, {dur}s\n")
    f.write(f"- logs: `{os.path.relpath(run_dir, repo_root)}/`\n\n")

# Update `latest` symlink (relative, so it's stable across mount points)
latest = os.path.join(pipeline_runs_dir, "latest")
try:
    if os.path.islink(latest) or os.path.exists(latest):
        os.unlink(latest)
    os.symlink(os.path.basename(run_dir), latest)
except Exception as e:
    print(f"[pipeline] warning: could not update 'latest' symlink: {e}", file=sys.stderr)
PY

    # Visible summary printed to terminal after everything.
    local rel_run_dir
    rel_run_dir="$(python3 -c "import os,sys; print(os.path.relpath(sys.argv[1], sys.argv[2]))" \
        "${RLCOMP_PIPELINE_RUN_DIR}" "${RLCOMP_REPO_ROOT}" 2>/dev/null \
        || echo "${RLCOMP_PIPELINE_RUN_DIR}")"
    echo ""
    echo "[pipeline] === summary ==="
    echo "[pipeline] exit_code       : ${exit_code}"
    echo "[pipeline] duration_sec    : ${duration}"
    echo "[pipeline] run logs        : ${rel_run_dir}/"
    echo "[pipeline] latest symlink  : results/pipeline_runs/latest -> $(basename "${RLCOMP_PIPELINE_RUN_DIR}")"
    echo "[pipeline] grep all runs   : jq -r '...' results/pipeline_runs/history.jsonl"
}

_run_logged() {
    # _run_logged <sub_action_name> <command...>
    # Captures stdout/stderr to per-step log files (streams to terminal too),
    # writes a <sub_action>.meta.json with timing + exit code.
    local sub_action="$1"; shift
    local stdout_log="${RLCOMP_PIPELINE_RUN_DIR}/${sub_action}.stdout.log"
    local stderr_log="${RLCOMP_PIPELINE_RUN_DIR}/${sub_action}.stderr.log"
    local meta_log="${RLCOMP_PIPELINE_RUN_DIR}/${sub_action}.meta.json"

    local tag start_iso start_epoch end_iso end_epoch duration exit_code
    if [ "${DRY_RUN}" = "1" ]; then tag='[dry-run]'; else tag='[run]'; fi
    printf '%s %s' "${tag}" "${sub_action}"
    printf ' %q' "$@"
    printf '\n'

    start_iso="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    start_epoch="$(date +%s)"
    exit_code=0
    if [ "${DRY_RUN}" = "1" ]; then
        echo "[dry-run] would execute: $*" | tee "${stdout_log}" >/dev/null
        : > "${stderr_log}"
    else
        # Stream stdout -> (terminal + stdout_log), stderr -> (terminal stderr + stderr_log).
        # Use process substitution (bash-only). Disable set -e around the call so
        # we can capture the exit code without the trap firing immediately.
        set +e
        "$@" > >(tee "${stdout_log}") 2> >(tee "${stderr_log}" >&2)
        exit_code=$?
        set -e
    fi
    end_iso="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    end_epoch="$(date +%s)"
    duration=$((end_epoch - start_epoch))

    # Write per-sub-action meta.json via python for JSON-safe escaping.
    "${RLCOMP_PYTHON}" - <<'PY' "${sub_action}" "$*" "${start_iso}" "${end_iso}" "${duration}" "${exit_code}" "${meta_log}" 2>/dev/null || true
import json, sys
sub_action, command, start, end, duration, exit_code, meta_path = sys.argv[1:8]
with open(meta_path, "w") as f:
    json.dump({
        "sub_action": sub_action,
        "command": command,
        "start_time_utc": start,
        "end_time_utc": end,
        "duration_sec": int(duration),
        "exit_code": int(exit_code),
    }, f, indent=2)
PY

    # Re-raise for set -e upstream so pipeline_finalize captures the real exit code.
    return "${exit_code}"
}

# ============================================================================
# action implementations
# ============================================================================

do_env() {
    _run_logged env bash -c '
        echo "[env] bootstrap already sourced. Banner was printed above."
        "'"${RLCOMP_PYTHON}"'" -c "import verl, vllm, ray, torch; print(f\"[env] verl/vllm/ray/torch import OK, cuda_avail={torch.cuda.is_available()}, n_gpu={torch.cuda.device_count()}\")"
    '
}

do_prepare() {
    _run_logged prepare bash scripts/experiments/prepare_newops_datasets.sh
}

do_stage1() {
    # run_stage1_newops_rft.sh reads these from env (${VAR:-default}).
    export PYTHON_BIN="${RLCOMP_PYTHON}"
    export TORCHRUN_BIN="${VIRTUAL_ENV}/bin/torchrun"
    _run_logged stage1 bash scripts/experiments/run_stage1_newops_rft.sh
}

do_stage2() {
    export PYTHON_BIN="${RLCOMP_PYTHON}"
    export CASE="${CASE:-newops}"
    export AUTO_RESUME="${AUTO_RESUME:-1}"
    export WANDB_ENABLE="${WANDB_ENABLE:-1}"
    local missing=0
    for f in \
        data/string_task/stage2_level1/train.parquet \
        data/string_task/stage2_level2/train.parquet \
        data/string_task/stage2_level1to8/forward_test.parquet \
        data/string_task/exp_new_ops/stage2_level1to8_newops_test.parquet; do
        if [ ! -f "${f}" ]; then
            echo "[run] stage2 needs missing dataset: ${f}" >&2
            missing=1
        fi
    done
    if [ "${missing}" -ne 0 ]; then
        echo "[run] run 'bash scripts/env/run.sh prepare' (plus stage2_level1/2 generators) first." >&2
        return 1
    fi
    _run_logged stage2 bash scripts/experiments/run_stage2_single_track.sh
}

do_infer() {
    export PYTHON_BIN="${RLCOMP_PYTHON}"
    _run_logged infer bash scripts/experiments/run_newops_inference_matrix.sh
}

do_all() {
    do_prepare
    do_stage1
    do_stage2
    do_infer
}

# ============================================================================
# dispatch
# ============================================================================

case "${ACTION}" in
    env|prepare|stage1|stage2|infer|all)
        _pipeline_init_logdir "$@"
        "do_${ACTION}" "$@"
        ;;
    -h|--help)
        sed -n '2,40p' "${_RUN_SRC}"
        ;;
    *)
        echo "Unknown action: ${ACTION}" >&2
        echo "Usage: bash scripts/env/run.sh <env|prepare|stage1|stage2|infer|all>" >&2
        exit 2
        ;;
esac
