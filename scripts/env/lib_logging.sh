#!/usr/bin/env bash
# ============================================================================
# RL-Compositionality — unified logging library (pure bash, no Python needed).
#
# Shared by the log viewer and by follow-up maintenance actions that do not go
# through run.sh. Produces a log directory with the same schema as run.sh's
# per-invocation output:
#
#   results/pipeline_runs/<ts>_<action>/
#       stdout.log
#       stderr.log
#       context.txt          (header: action, ts, argv, pwd, host, user, git)
#       meta.json            (finalizer output; JSON)
#   results/pipeline_runs/latest                  -> <ts>_<action>     (symlink)
#   results/pipeline_runs/history.jsonl           (one line per invocation)
#
# Exposed functions:
#   rlcomp_log_init  <action> <argv...>
#   rlcomp_log_finalize <exit_code>
#   rlcomp_log_wrap  <action> <cmd...>   # init + tee-run + finalize
#
# Portability contract: pure bash + coreutils (date, hostname, tee, ln, mkdir,
# printf, grep, awk, mktemp). No python. No jq. Works with `set -euo pipefail`
# in the caller, and does NOT leak those flags to the caller's shell.
# ============================================================================

# Double-source guard.
if [ -n "${_RLCOMP_LIB_LOGGING_LOADED:-}" ]; then
    return 0 2>/dev/null || true
fi
_RLCOMP_LIB_LOGGING_LOADED=1

# ---------- locate repo root once ---------------------------------------------
_LIB_LOGGING_SRC="${BASH_SOURCE[0]}"
RLCOMP_REPO_ROOT="${RLCOMP_REPO_ROOT:-$(cd "$(dirname "${_LIB_LOGGING_SRC}")/../.." && pwd)}"

# ---------- JSON escape (pure bash) -------------------------------------------
# Escapes a single string for embedding inside a JSON string literal.
# Handles: \  "  \n  \r  \t  \b  \f  and control chars 0x00-0x1f.
_rlcomp_json_escape() {
    local s="$1"
    local out="" i ch code
    for ((i = 0; i < ${#s}; i++)); do
        ch="${s:i:1}"
        case "${ch}" in
            '\') out+='\\' ;;
            '"') out+='\"' ;;
            $'\n') out+='\n' ;;
            $'\r') out+='\r' ;;
            $'\t') out+='\t' ;;
            $'\b') out+='\b' ;;
            $'\f') out+='\f' ;;
            *)
                printf -v code '%d' "'${ch}"
                if [ "${code}" -lt 32 ]; then
                    printf -v out '%s\\u%04x' "${out}" "${code}"
                else
                    out+="${ch}"
                fi
                ;;
        esac
    done
    printf '%s' "${out}"
}

# ---------- compute a writable pipeline_runs root -----------------------------
# Prefers repo/results/pipeline_runs; if that dir exists but is un-writable
# (root-owned), falls back to repo/logs/pipeline_runs. Returns path on stdout.
_rlcomp_log_runs_root() {
    local preferred="${RLCOMP_REPO_ROOT}/results/pipeline_runs"
    local fallback="${RLCOMP_REPO_ROOT}/logs/pipeline_runs"
    local parent
    parent="$(dirname "${preferred}")"
    # Ensure results/ exists (creating logs/ fallback later if this fails).
    if mkdir -p "${preferred}" 2>/dev/null && [ -w "${preferred}" ]; then
        printf '%s\n' "${preferred}"
        return 0
    fi
    # Some other process (root) created it. Try a probe write.
    if [ -d "${preferred}" ] && [ -w "${preferred}" ]; then
        printf '%s\n' "${preferred}"
        return 0
    fi
    # Fall back into logs/ (repo root is typically writable by ma-user).
    if mkdir -p "${fallback}" 2>/dev/null && [ -w "${fallback}" ]; then
        printf '%s\n' "${fallback}"
        return 0
    fi
    printf '[rlcomp] ERROR: no writable repo-local log directory under %s\n' "${RLCOMP_REPO_ROOT}" >&2
    return 1
}

# ---------- rlcomp_log_init <action> <argv...> --------------------------------
# Sets RLCOMP_LOG_DIR, RLCOMP_LOG_TS, RLCOMP_LOG_START_EPOCH, RLCOMP_LOG_ACTION,
# RLCOMP_LOG_ARGV_STR. Writes context.txt. Echoes the log dir to stdout.
rlcomp_log_init() {
    local action="$1"; shift
    local runs_root
    runs_root="$(_rlcomp_log_runs_root)"
    local ts start_iso start_epoch log_dir host user git_commit pwd_s
    start_iso="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    start_epoch="$(date +%s)"
    # Reuse the parent wrapper's run dir if rlcomp.sh already set one up. This
    # keeps the outer terminal mirror and this wrapper's meta.json in the same
    # directory, instead of nesting a second timestamped dir one level deeper.
    if [ -n "${RLCOMP_PIPELINE_RUN_DIR:-}" ] && [ -d "${RLCOMP_PIPELINE_RUN_DIR}" ]; then
        log_dir="${RLCOMP_PIPELINE_RUN_DIR}"
        ts="${RLCOMP_PIPELINE_TS:-$(date +%Y%m%d_%H%M%S)}"
        runs_root="$(dirname "${log_dir}")"
    else
        ts="$(date +%Y%m%d_%H%M%S)"
        log_dir="${runs_root}/${ts}_${action}"
        # In the very unlikely event of a timestamp clash, append a counter.
        local dup=1
        while [ -e "${log_dir}" ]; do
            log_dir="${runs_root}/${ts}_${action}_${dup}"
            dup=$((dup + 1))
        done
        mkdir -p "${log_dir}"
    fi

    host="$(hostname 2>/dev/null || echo unknown)"
    user="${USER:-$(id -un 2>/dev/null || echo unknown)}"
    pwd_s="$(pwd 2>/dev/null || echo unknown)"
    git_commit="$(git -C "${RLCOMP_REPO_ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)"

    # Build a shell-safe argv string for the context header.
    local argv_str=""
    if [ "$#" -gt 0 ]; then
        printf -v argv_str ' %q' "$@"
    fi

    {
        printf 'action: %s\n' "${action}"
        printf 'ts: %s\n' "${ts}"
        printf 'start_utc: %s\n' "${start_iso}"
        printf 'run_dir: %s\n' "${log_dir}"
        printf 'argv:%s\n' "${argv_str}"
        printf 'pwd: %s\n' "${pwd_s}"
        printf 'user: %s@%s\n' "${user}" "${host}"
        printf 'git_commit: %s\n' "${git_commit}"
    } > "${log_dir}/context.txt" 2>/dev/null || true

    export RLCOMP_LOG_DIR="${log_dir}"
    export RLCOMP_LOG_TS="${ts}"
    export RLCOMP_LOG_START_EPOCH="${start_epoch}"
    export RLCOMP_LOG_START_ISO="${start_iso}"
    export RLCOMP_LOG_ACTION="${action}"
    export RLCOMP_LOG_ARGV_STR="${argv_str}"
    export RLCOMP_LOG_HOST="${host}"
    export RLCOMP_LOG_USER="${user}"
    export RLCOMP_LOG_PWD="${pwd_s}"
    export RLCOMP_LOG_GIT_COMMIT="${git_commit}"
    export RLCOMP_LOG_RUNS_ROOT="${runs_root}"

    printf '%s\n' "${log_dir}"
}

# ---------- rlcomp_log_finalize <exit_code> -----------------------------------
# Writes meta.json, updates latest symlink, appends to history.jsonl,
# prints a short summary to stderr.
rlcomp_log_finalize() {
    local exit_code="${1:-0}"
    # If init never ran (programming error), bail quietly.
    if [ -z "${RLCOMP_LOG_DIR:-}" ]; then
        return 0
    fi
    local end_iso end_epoch duration
    end_iso="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    end_epoch="$(date +%s)"
    duration=$((end_epoch - RLCOMP_LOG_START_EPOCH))

    local action_e start_e end_e host_e git_e argv_e user_e pwd_e runid_e
    action_e="$(_rlcomp_json_escape "${RLCOMP_LOG_ACTION}")"
    start_e="$(_rlcomp_json_escape "${RLCOMP_LOG_START_ISO}")"
    end_e="$(_rlcomp_json_escape "${end_iso}")"
    host_e="$(_rlcomp_json_escape "${RLCOMP_LOG_HOST}")"
    git_e="$(_rlcomp_json_escape "${RLCOMP_LOG_GIT_COMMIT}")"
    argv_e="$(_rlcomp_json_escape "${RLCOMP_LOG_ARGV_STR}")"
    user_e="$(_rlcomp_json_escape "${RLCOMP_LOG_USER}")"
    pwd_e="$(_rlcomp_json_escape "${RLCOMP_LOG_PWD}")"
    runid_e="$(_rlcomp_json_escape "$(basename "${RLCOMP_LOG_DIR}")")"

    # Write meta.json via heredoc (no python, no jq).
    cat > "${RLCOMP_LOG_DIR}/meta.json" <<EOF
{
  "run_id": "${runid_e}",
  "action": "${action_e}",
  "start_time_utc": "${start_e}",
  "end_time_utc": "${end_e}",
  "duration_sec": ${duration},
  "exit_code": ${exit_code},
  "host": "${host_e}",
  "user": "${user_e}",
  "pwd": "${pwd_e}",
  "git_commit": "${git_e}",
  "argv": "${argv_e}",
  "wrapper": "lib_logging.sh"
}
EOF

    # Update `latest` symlink (relative, atomic-ish via mktemp + mv).
    local runs_root latest_link latest_tmp
    runs_root="${RLCOMP_LOG_RUNS_ROOT}"
    latest_link="${runs_root}/latest"
    latest_tmp="${runs_root}/.latest.$$"
    ln -sfn "$(basename "${RLCOMP_LOG_DIR}")" "${latest_tmp}" 2>/dev/null && \
        mv -Tf "${latest_tmp}" "${latest_link}" 2>/dev/null || {
            # `mv -T` not available everywhere; fall back to rm+ln.
            rm -f "${latest_link}" 2>/dev/null
            ln -sfn "$(basename "${RLCOMP_LOG_DIR}")" "${latest_link}" 2>/dev/null || true
            rm -f "${latest_tmp}" 2>/dev/null || true
        }

    # Append a compact history.jsonl line.
    local hist_line
    hist_line=$(cat <<EOF
{"run_id": "${runid_e}", "action": "${action_e}", "start_time_utc": "${start_e}", "end_time_utc": "${end_e}", "duration_sec": ${duration}, "exit_code": ${exit_code}, "host": "${host_e}", "git_commit": "${git_e}", "wrapper": "lib_logging.sh"}
EOF
)
    printf '%s\n' "${hist_line}" >> "${runs_root}/history.jsonl" 2>/dev/null || true

    # Human-readable summary to stderr.
    local rel
    rel="${RLCOMP_LOG_DIR#${RLCOMP_REPO_ROOT}/}"
    {
        printf '[rlcomp] === log summary ===\n'
        printf '[rlcomp] action      : %s\n' "${RLCOMP_LOG_ACTION}"
        printf '[rlcomp] exit_code   : %s\n' "${exit_code}"
        printf '[rlcomp] duration_sec: %s\n' "${duration}"
        printf '[rlcomp] log dir     : %s\n' "${rel}"
        printf '[rlcomp] latest link : %s -> %s\n' \
            "${runs_root#${RLCOMP_REPO_ROOT}/}/latest" \
            "$(basename "${RLCOMP_LOG_DIR}")"
    } >&2
}

# ---------- rlcomp_log_wrap <action> <cmd...> ---------------------------------
# One-shot: init + execute with stdout/stderr tee'd + finalize.
# - stdout of the wrapped command goes to BOTH the terminal's stdout AND
#   ${RLCOMP_LOG_DIR}/stdout.log in real time.
# - stderr of the wrapped command goes to BOTH the terminal's stderr AND
#   ${RLCOMP_LOG_DIR}/stderr.log in real time.
# - Exit code of the wrapped command is preserved and returned.
rlcomp_log_wrap() {
    local action="$1"; shift
    # _log_init prints the dir; capture without spamming.
    rlcomp_log_init "${action}" "$@" >/dev/null

    local exit_code=0
    local stdout_log="${RLCOMP_LOG_DIR}/stdout.log"
    local stderr_log="${RLCOMP_LOG_DIR}/stderr.log"

    local _had_errexit=0
    case $- in *e*) _had_errexit=1 ;; esac
    set +e
    if [ -n "${RLCOMP_PIPELINE_RUN_DIR:-}" ] && \
       [ "${RLCOMP_PIPELINE_RUN_DIR}" = "${RLCOMP_LOG_DIR}" ]; then
        # rlcomp.sh has already installed an outer terminal mirror that captures
        # every byte of this shell's stdout/stderr to terminal.{stdout,stderr}.log
        # in the SAME run_dir. Wrapping the command with another tee would just
        # duplicate every line into stdout.log — drop the inner tee and let the
        # outer mirror do the capture.
        "$@"
        exit_code=$?
    else
        # Standalone use (no outer mirror): tee the command's output to both
        # the terminal and per-call stdout.log / stderr.log.
        "$@" > >(tee -a "${stdout_log}") 2> >(tee -a "${stderr_log}" >&2)
        exit_code=$?
        wait 2>/dev/null || true
    fi
    [ "${_had_errexit}" -eq 1 ] && set -e

    rlcomp_log_finalize "${exit_code}"
    return "${exit_code}"
}

# ---------- utility: resolve a log dir by id or "latest" ----------------------
# Used by logs.sh.
rlcomp_log_resolve_dir() {
    local id="${1:-latest}"
    local runs_root
    runs_root="$(_rlcomp_log_runs_root)"
    if [ "${id}" = "latest" ]; then
        if [ -e "${runs_root}/latest" ]; then
            # Symlink or dir; both work with readlink -f then basename fallback.
            local target
            target="$(readlink "${runs_root}/latest" 2>/dev/null || true)"
            if [ -n "${target}" ]; then
                case "${target}" in
                    /*) printf '%s\n' "${target}" ;;
                    *)  printf '%s/%s\n' "${runs_root}" "${target}" ;;
                esac
                return 0
            fi
            printf '%s\n' "${runs_root}/latest"
            return 0
        fi
        echo "[rlcomp] no runs under ${runs_root}" >&2
        return 1
    fi
    if [ -d "${runs_root}/${id}" ]; then
        printf '%s\n' "${runs_root}/${id}"
        return 0
    fi
    echo "[rlcomp] no such run: ${id}" >&2
    return 1
}
