#!/usr/bin/env bash
# ============================================================================
# RL-Compositionality — pipeline log viewer.
#
# Usage (from rlcomp.sh):
#   rlcomp.sh logs              # show latest run's meta + paths
#   rlcomp.sh logs peek [N]     # last N lines (default 30) of the most-active run
#   rlcomp.sh logs <N>          # shortcut: same as `peek N`  (e.g.  logs 50)
#   rlcomp.sh logs tail         # tail -F latest stdout+stderr (live follow)
#   rlcomp.sh logs list         # last 10 entries from history.jsonl
#   rlcomp.sh logs cat          # cat latest stdout + stderr
#   rlcomp.sh logs show <id>    # cat a specific run's logs (id = <ts>_<action>)
#
# `peek` differs from `tail` two ways:
#   1. one-shot (no -F follow); good for a quick glance and gone
#   2. picks the *most recently modified* run dir, not the `latest` symlink.
#      The symlink only updates on _pipeline_finalize, so during an active
#      run it points at the previous (finished) run. peek picks by mtime so
#      you always see the run that's actually writing logs right now.
# ============================================================================
set -uo pipefail

_LOGS_SRC="${BASH_SOURCE[0]}"
_LOGS_DIR="$(cd "$(dirname "${_LOGS_SRC}")" && pwd)"

# shellcheck disable=SC1091
. "${_LOGS_DIR}/lib_logging.sh"

SUB="${1:-show-latest}"
shift || true

# Locate runs root. lib_logging exposes a private helper; replicate the simple
# logic here because we want to work from a cold shell (no prior init).
_logs_runs_root() {
    local preferred="${RLCOMP_REPO_ROOT}/results/pipeline_runs"
    local fallback="${RLCOMP_REPO_ROOT}/logs/pipeline_runs"
    if [ -d "${preferred}" ] && [ -r "${preferred}" ]; then
        printf '%s\n' "${preferred}"
        return 0
    fi
    if [ -d "${fallback}" ] && [ -r "${fallback}" ]; then
        printf '%s\n' "${fallback}"
        return 0
    fi
    # Default to preferred even if missing — callers will report "no runs".
    printf '%s\n' "${preferred}"
}

_logs_has_jq() {
    command -v jq >/dev/null 2>&1
}

_logs_show_latest_meta() {
    local dir="$1"
    local meta inv ctx
    meta="${dir}/meta.json"
    inv="${dir}/invocation.meta.json"
    ctx="${dir}/context.txt"

    printf '[logs] run dir: %s\n' "${dir}"
    if [ -f "${meta}" ]; then
        printf '\n--- meta.json ---\n'
        if _logs_has_jq; then
            jq . "${meta}"
        else
            cat "${meta}"
        fi
    fi
    if [ -f "${inv}" ]; then
        printf '\n--- invocation.meta.json (nested run.sh log) ---\n'
        if _logs_has_jq; then
            jq '{run_id, action, start_time_utc, end_time_utc, duration_sec, exit_code, host, git_commit, sub_actions: [.sub_actions[]? | {sub_action, exit_code, duration_sec}]}' "${inv}"
        else
            cat "${inv}"
        fi
    fi
    if [ -f "${ctx}" ]; then
        printf '\n--- context.txt ---\n'
        cat "${ctx}"
    fi
    printf '\n--- log files ---\n'
    # List every *.log file with size.
    local f
    for f in "${dir}"/*.log; do
        [ -e "${f}" ] || continue
        printf '  %s  (%s)\n' "${f}" "$(du -h "${f}" 2>/dev/null | awk '{print $1}')"
    done
}

_logs_tail_latest() {
    local dir="$1"
    # Prefer top-level stdout.log/stderr.log (from lib_logging wrapper); fall
    # back to per-step logs (run.sh style, e.g. env.stdout.log).
    local stdout_log="${dir}/stdout.log"
    local stderr_log="${dir}/stderr.log"
    if [ ! -f "${stdout_log}" ]; then
        stdout_log="$(ls "${dir}"/*.stdout.log 2>/dev/null | tail -1)"
    fi
    if [ ! -f "${stderr_log}" ]; then
        stderr_log="$(ls "${dir}"/*.stderr.log 2>/dev/null | tail -1)"
    fi
    if [ -z "${stdout_log:-}" ] && [ -z "${stderr_log:-}" ]; then
        echo "[logs] no log files in ${dir}" >&2
        return 1
    fi
    # Two cases:
    #   - both present -> tail -f both (stderr lines are tagged by tail's header)
    #   - only one     -> tail -f that one
    if [ -f "${stdout_log:-}" ] && [ -f "${stderr_log:-}" ]; then
        echo "[logs] tail -f ${stdout_log} ${stderr_log}  (Ctrl+C to stop)" >&2
        tail -n 50 -F "${stdout_log}" "${stderr_log}"
    elif [ -f "${stdout_log:-}" ]; then
        echo "[logs] tail -f ${stdout_log}  (Ctrl+C to stop)" >&2
        tail -n 50 -F "${stdout_log}"
    else
        echo "[logs] tail -f ${stderr_log}  (Ctrl+C to stop)" >&2
        tail -n 50 -F "${stderr_log}"
    fi
}

_logs_cat_run() {
    local dir="$1"
    local f
    local any=0
    for f in "${dir}"/stdout.log "${dir}"/*.stdout.log; do
        [ -e "${f}" ] || continue
        printf '\n===== %s =====\n' "${f}"
        cat "${f}"
        any=1
    done
    for f in "${dir}"/stderr.log "${dir}"/*.stderr.log; do
        [ -e "${f}" ] || continue
        printf '\n===== %s =====\n' "${f}"
        cat "${f}"
        any=1
    done
    if [ "${any}" -eq 0 ]; then
        echo "[logs] no log files in ${dir}" >&2
        return 1
    fi
}

_logs_list_history() {
    local runs_root="$1"
    local hist="${runs_root}/history.jsonl"
    if [ ! -f "${hist}" ]; then
        echo "[logs] no history.jsonl under ${runs_root}" >&2
        return 0
    fi
    local n="${1:-10}"
    # We'll print the last 10 entries. `tail -n 10 file` is universal.
    if _logs_has_jq; then
        printf '%-22s %-10s %6s %4s %-12s %s\n' \
            "RUN_ID" "ACTION" "DUR_S" "EXIT" "HOST" "GIT"
        printf '%s\n' "-------------------------------------------------------------------------------"
        tail -n 10 "${hist}" | jq -r '[.run_id, .action, (.duration_sec|tostring), (.exit_code|tostring), .host, (.git_commit[0:10])] | @tsv' \
            | awk -F'\t' '{printf "%-22s %-10s %6s %4s %-12s %s\n", $1, $2, $3, $4, $5, $6}'
    else
        # Fallback: pull fields via sed. Good enough for the top-level fields
        # we always write (no nested objects to worry about).
        printf '%-22s %-10s %6s %4s %-12s %s\n' \
            "RUN_ID" "ACTION" "DUR_S" "EXIT" "HOST" "GIT"
        printf '%s\n' "-------------------------------------------------------------------------------"
        tail -n 10 "${hist}" | while IFS= read -r line; do
            rid=$(printf '%s' "${line}" | sed -n 's/.*"run_id": *"\([^"]*\)".*/\1/p')
            act=$(printf '%s' "${line}" | sed -n 's/.*"action": *"\([^"]*\)".*/\1/p')
            dur=$(printf '%s' "${line}" | sed -n 's/.*"duration_sec": *\([0-9]*\).*/\1/p')
            ex=$(printf '%s' "${line}" | sed -n 's/.*"exit_code": *\([0-9-]*\).*/\1/p')
            host=$(printf '%s' "${line}" | sed -n 's/.*"host": *"\([^"]*\)".*/\1/p')
            gc=$(printf '%s' "${line}" | sed -n 's/.*"git_commit": *"\([^"]*\)".*/\1/p')
            printf '%-22s %-10s %6s %4s %-12s %s\n' \
                "${rid}" "${act}" "${dur}" "${ex}" "${host}" "${gc:0:10}"
        done
    fi
}

# Resolve the run dir most likely to be of interest to a "what's happening
# right now" query. Strategy: pick the run dir with the freshest mtime in
# pipeline_runs/. This intentionally bypasses the `latest` symlink because
# `latest` only updates on _pipeline_finalize — i.e., during an in-flight
# run it points at the previous (finished) run, which is NOT what you want
# when monitoring progress.
_logs_resolve_active() {
    local runs_root="$1"
    # Use stat-by-mtime (-printf '%T@ %p\n') to sort numerically, then pick
    # the freshest. Falls back to ls -t if find -printf isn't available
    # (e.g. on macOS / BSD).
    local newest
    newest="$(find "${runs_root}" -mindepth 1 -maxdepth 1 -type d \
              -printf '%T@\t%p\n' 2>/dev/null \
              | sort -nr | head -1 | awk -F'\t' '{print $2}')"
    if [ -z "${newest}" ]; then
        newest="$(ls -1td "${runs_root}"/*_*/ 2>/dev/null | head -1)"
        # Strip trailing slash from ls -d output for consistency
        newest="${newest%/}"
    fi
    [ -n "${newest}" ] && [ -d "${newest}" ] && printf '%s\n' "${newest}"
}

# One-shot tail of the active run's logs. No follow. Picks both top-level
# stdout/stderr (rlcomp.sh outer-tee + lib_logging wrapper) AND every
# per-sub-action *.stdout.log / *.stderr.log emitted by run.sh, prints
# their last N lines so you see the freshest progress regardless of which
# layer is writing.
_logs_peek() {
    local dir="$1"
    local n="${2:-30}"
    local rel="${dir#${RLCOMP_REPO_ROOT:-/}/}"
    printf '[logs peek] dir : %s  (mtime %s)\n' "${rel}" \
        "$(date -r "${dir}" +'%Y-%m-%d %H:%M:%S' 2>/dev/null || echo '?')"
    printf '[logs peek] last : %s lines per file\n\n' "${n}"
    local f shown=0
    # Order matters: most useful first. terminal.* (outer rlcomp.sh tee)
    # has bootstrap warnings + summary; sub-action *.stdout/stderr.log have
    # the actual training/rollout progress. Both are interesting.
    for f in "${dir}"/terminal.stdout.log "${dir}"/terminal.stderr.log \
             "${dir}"/stdout.log "${dir}"/stderr.log \
             "${dir}"/*.stdout.log "${dir}"/*.stderr.log; do
        [ -e "${f}" ] || continue
        # Skip the per-sub-action file if its name collides with a top-level
        # file we already printed (avoid duplicate output).
        case "${f}" in
            "${dir}/stdout.log"|"${dir}/stderr.log"|\
            "${dir}/terminal.stdout.log"|"${dir}/terminal.stderr.log") continue ;;
            *)
                # filter out files starting with terminal. or named stdout/stderr
                # (already printed above)
                ;;
        esac
        local sz
        sz="$(stat -c '%s' "${f}" 2>/dev/null || echo '?')"
        printf '===== %s  (%sB) — last %s lines =====\n' \
            "${f##*/}" "${sz}" "${n}"
        tail -n "${n}" "${f}" 2>/dev/null
        printf '\n'
        shown=$((shown + 1))
    done
    if [ "${shown}" -eq 0 ]; then
        echo "[logs peek] no log files in ${dir}" >&2
        return 1
    fi
}

runs_root="$(_logs_runs_root)"

case "${SUB}" in
    show-latest|"")
        if dir="$(rlcomp_log_resolve_dir latest 2>/dev/null)" && [ -n "${dir}" ]; then
            _logs_show_latest_meta "${dir}"
        else
            echo "[logs] no runs under ${runs_root}" >&2
            exit 1
        fi
        ;;
    peek)
        N="${1:-30}"
        if dir="$(_logs_resolve_active "${runs_root}")" && [ -n "${dir}" ]; then
            _logs_peek "${dir}" "${N}"
        else
            echo "[logs] no runs under ${runs_root}" >&2
            exit 1
        fi
        ;;
    [0-9]*)
        # Shortcut: `rlcomp.sh logs 50` == `rlcomp.sh logs peek 50`.
        if dir="$(_logs_resolve_active "${runs_root}")" && [ -n "${dir}" ]; then
            _logs_peek "${dir}" "${SUB}"
        else
            echo "[logs] no runs under ${runs_root}" >&2
            exit 1
        fi
        ;;
    tail)
        if dir="$(rlcomp_log_resolve_dir latest 2>/dev/null)" && [ -n "${dir}" ]; then
            _logs_tail_latest "${dir}"
        else
            echo "[logs] no runs under ${runs_root}" >&2
            exit 1
        fi
        ;;
    list)
        _logs_list_history "${runs_root}"
        ;;
    cat)
        if dir="$(rlcomp_log_resolve_dir latest 2>/dev/null)" && [ -n "${dir}" ]; then
            _logs_cat_run "${dir}"
        else
            echo "[logs] no runs under ${runs_root}" >&2
            exit 1
        fi
        ;;
    show)
        ID="${1:-}"
        if [ -z "${ID}" ]; then
            echo "[logs] usage: rlcomp.sh logs show <ts>_<action>" >&2
            exit 2
        fi
        if dir="$(rlcomp_log_resolve_dir "${ID}" 2>/dev/null)" && [ -n "${dir}" ]; then
            _logs_show_latest_meta "${dir}"
            printf '\n'
            _logs_cat_run "${dir}"
        else
            echo "[logs] no such run: ${ID}" >&2
            exit 1
        fi
        ;;
    -h|--help)
        sed -n '2,21p' "${_LOGS_SRC}"
        ;;
    *)
        echo "[logs] unknown subcommand: ${SUB}" >&2
        echo "[logs] valid: (no arg) | peek [N] | <N> | tail | list | cat | show <id>" >&2
        exit 2
        ;;
esac
