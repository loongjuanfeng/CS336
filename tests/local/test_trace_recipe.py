import os
import shutil
import subprocess
from pathlib import Path


def test_trace_recipe(tmp_path):
    shutil.copyfile(Path(__file__).parents[2] / "justfile", tmp_path / "justfile")
    uv = tmp_path / "uv"
    uv.write_text(
        "#!/usr/bin/env bash\n"
        "set -eu\n"
        'if [[ "$1" == run && "$2" == nsys && "$3" == profile ]]; then\n'
        '    [[ "$TORCHINDUCTOR_COMPILE_THREADS" == 1 ]]\n'
        '    for arg in "$@"; do\n'
        '        case "$arg" in --output=*) touch "${arg#--output=}.nsys-rep";; esac\n'
        "    done\n"
        '    printf "<%s>\\n" "$@"\n'
        "else\n"
        "    exit 1\n"
        "fi\n"
    )
    uv.chmod(0o755)
    uvx = tmp_path / "uvx"
    uvx.write_text(
        "#!/usr/bin/env bash\n"
        "set -eu\n"
        '[[ "$1" == ncompass && "$2" == convert && "$4" == --quiet ]]\n'
        'test -f "$3"\n'
        'touch "${3%.nsys-rep}.json.gz"\n'
    )
    uvx.chmod(0o755)
    traces = tmp_path / "results/traces"
    traces.mkdir(parents=True)
    (traces / "trace-1.nsys-rep").write_text("existing report")
    (traces / "trace-3.json.gz").write_text("existing export")
    for number in (2, 4):
        result = subprocess.run(
            ["just", "trace", "script with spaces.py", "--steps", "1"],
            cwd=tmp_path,
            env={**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}",
                 "TORCHINDUCTOR_COMPILE_THREADS": ""},
            capture_output=True,
            text=True,
            check=True,
        )
        assert "<script with spaces.py>\n<--steps>\n<1>" in result.stdout
        options = [line[1:-1] for line in result.stdout.splitlines() if line.startswith("<--")]
        assert options == [
            f"--output=results/traces/trace-{number}",
            "--trace=cuda,nvtx",
            "--sample=none",
            "--cpuctxsw=none",
            "--stats=false",
            "--steps",
        ]
        assert (traces / f"trace-{number}.nsys-rep").is_file()
        assert (traces / f"trace-{number}.json.gz").is_file()
        assert f"Perfetto: results/traces/trace-{number}.json.gz" in result.stdout
    assert not list(traces.glob("*.log"))
    assert (traces / "trace-1.nsys-rep").read_text() == "existing report"
    assert (traces / "trace-3.json.gz").read_text() == "existing export"
