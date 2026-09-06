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

# Completed A1 baseline (also the default pytest selection).
test:
    uv run pytest -q

test-local:
    uv run pytest tests/local -q

test-a1:
    uv run pytest tests/assignment1 -q

# A2 remains unimplemented; use collection to verify the starter wiring.
collect-a2:
    uv run pytest tests/assignment2 --collect-only -q

test-a2:
    uv run pytest tests/assignment2 -q

smoke:
    uv run cs336-train --smoke --steps 10 --output-dir checkpoints/smoke
