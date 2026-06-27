from __future__ import annotations

import subprocess
import sys


def test_pipeline_run_dry_run() -> None:
    cmd = [
        sys.executable,
        "apps/run_empirical.py",
        "pipeline-run",
        "--",
        "--pair",
        "rlusd_xrp",
        "--ledger-start",
        "1",
        "--ledger-end",
        "2",
        "--tx-prebook-snapshots",
        "dummy_tx_prebook.ndjson",
        "--metadata-ndjson",
        "dummy_metadata.ndjson",
        "--dry-run",
    ]
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "[dry-run] planned pipeline steps:" in result.stdout
    assert "empirical_export_window.py" in result.stdout
    assert "empirical_build_model_input.py" in result.stdout
    assert "empirical_compare_rolling.py" in result.stdout
