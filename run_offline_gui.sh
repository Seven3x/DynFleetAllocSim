#!/usr/bin/env bash
set -euo pipefail

SCENARIO_FILE="examples/scenario_verification_demo.json"
LOG_FILE=""
DEBUG_LOG=1

usage() {
    cat <<'EOF'
Usage: run_offline_gui.sh [options]

Options:
  -s, --scenario-file <path>  Scenario file path (default: examples/scenario_verification_demo.json)
  -l, --log-file <path>       Log file path (default with debug: outputs/offline_gui_debug_<timestamp>.log)
      --debug-log             Enable debug log (default)
      --no-debug-log          Disable debug log
  -h, --help                  Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -s|--scenario-file)
            if [[ $# -lt 2 ]]; then
                echo "Missing value for $1" >&2
                usage >&2
                exit 2
            fi
            SCENARIO_FILE="$2"
            shift 2
            ;;
        -l|--log-file)
            if [[ $# -lt 2 ]]; then
                echo "Missing value for $1" >&2
                usage >&2
                exit 2
            fi
            LOG_FILE="$2"
            shift 2
            ;;
        --debug-log)
            DEBUG_LOG=1
            shift
            ;;
        --no-debug-log)
            DEBUG_LOG=0
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

export PYTHONPATH="src"

if [[ "${DEBUG_LOG}" -eq 1 ]]; then
    export MILP_DEBUG="1"
    if [[ -z "${LOG_FILE}" ]]; then
        stamp="$(date +"%Y%m%d_%H%M%S")"
        LOG_FILE="outputs/offline_gui_debug_${stamp}.log"
    fi
fi

args=("-u" "-m" "milp_sim.main" "--gui")
if [[ -n "${SCENARIO_FILE}" ]]; then
    args+=("--scenario-file" "${SCENARIO_FILE}")
fi

exit_code=0
if [[ -n "${LOG_FILE}" ]]; then
    log_dir="$(dirname "${LOG_FILE}")"
    if [[ -n "${log_dir}" && "${log_dir}" != "." ]]; then
        mkdir -p "${log_dir}"
    fi
    echo "Logging to ${LOG_FILE}"
    set +e
    python "${args[@]}" 2>&1 | tee "${LOG_FILE}"
    exit_code="${PIPESTATUS[0]}"
    set -e
else
    set +e
    python "${args[@]}"
    exit_code="$?"
    set -e
fi

if [[ "${exit_code}" -ne 0 ]]; then
    echo "python exited with code ${exit_code}" >&2
    exit "${exit_code}"
fi
