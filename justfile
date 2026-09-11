check:
    uv run ruff check src examples scripts tests/local
    uv run ty check

format:
    uv run ruff format src tests/local

environment mode:
    #!/usr/bin/env bash
    set -euo pipefail

    case "{{mode}}" in
        local)
            env -u UV_NO_SOURCES \
                UV_PROJECT_ENVIRONMENT="$PWD/.venvs/local" \
                uv sync
            ;;
        default)
            UV_PROJECT_ENVIRONMENT="$PWD/.venvs/default" \
                uv sync --no-sources
            ;;
        *)
            echo "usage: just environment <local|default>" >&2
            exit 2
            ;;
    esac

    if [[ -e .venv && ! -L .venv ]]; then
        echo "refusing to replace existing .venv file or directory" >&2
        exit 1
    fi

    ln -sfnT ".venvs/{{mode}}" .venv
    printf 'CS336_ENVIRONMENT=%s\n' "{{mode}}" > .env

test:
    uv run pytest -q

test-local:
    uv run pytest tests/local -q

test-a1:
    uv run pytest tests/assignment1 -q

test-a2:
    uv run pytest tests/assignment2 -q

smoke:
    uv run cs336-train --smoke --steps 10 --output-dir checkpoints/smoke

# CUDA kernels + the script's NVTX regions. Extra arguments go to the script.
[positional-arguments]
@trace script *args:
    #!/usr/bin/env bash
    set -euo pipefail
    mkdir -p results/traces
    run_id=1
    while [[ -e "results/traces/trace-$run_id.nsys-rep" || -e "results/traces/trace-$run_id.json.gz" ]]; do
        run_id=$((run_id + 1))
    done
    output="results/traces/trace-$run_id"
    export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"
    uv run nsys profile \
        --output="$output" \
        --trace=cuda,nvtx \
        --sample=none \
        --cpuctxsw=none \
        --stats=false \
        python "$@"
    # Chrome trace JSON is readable by Perfetto; keep the original Nsight report.
    uvx ncompass convert "$output.nsys-rep" --quiet
    printf 'Nsight: %s.nsys-rep\nPerfetto: %s.json.gz\n' "$output" "$output"
