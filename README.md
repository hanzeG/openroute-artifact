# OpenRoute Artifact

This repository contains the lightweight code artifact for OpenRoute. It is
intended for anonymous review and contains the implementation and scripts needed
to reproduce the data-processing path, replay checks, and same-fill routing
optimisation experiments.

The paper source, paper build outputs, full XRPL ledger snapshots, and large
intermediate experiment artifacts are intentionally not included.

## Repository Contents

- `src/xrpl_router/`: AMM/CLOB amount domains, executable source kernels, route
  steps, transaction intent handling, and replay/optimisation primitives.
- `empirical/scripts/`: data-fetching, pre-state construction, replay,
  comparison, optimisation, and summarisation scripts.
- `empirical/rust/tx_prebook_replay/`: Rust transaction-prebook replay used for
  large replay workloads.
- `configs/empirical/`: example pipeline configuration.
- `data/config.share.example`: template for local RPC/share credentials.
- `tests/`: unit, rebuild, and smoke tests for the routing and empirical code.

## Not Included

The full experimental run uses large XRPL metadata, account-state snapshots,
transaction-prebook replay outputs, and pair-level intermediate files. These
files are omitted because they are large generated artifacts. The repository
provides the scripts and configuration path for regenerating them.

The paper PDF is submitted separately through the review system.

## Environment

Create the Python environment:

```bash
conda env create -f environment.yml
conda activate xrpl-amm-clob
pip install -e .
```

The Rust replay helper requires a local Rust toolchain:

```bash
cd empirical/rust/tx_prebook_replay
cargo build --release
```

## Quick Checks

Run lightweight checks that do not require the full XRPL dataset:

```bash
pytest tests/ci/test_smoke_imports.py
pytest tests/rebuild/test_core_kernel.py tests/rebuild/test_amm_rebuild.py
```

## Full Reproduction Path

The full workflow is data-intensive. At a high level:

1. Configure local XRPL/RPC access from `data/config.share.example`.
2. Fetch transaction metadata and ledger-state snapshots for the target ledger
   window.
3. Construct direct XRP-token targets and transaction-level pre-state inputs.
4. Replay transaction prebooks with the Rust replay helper.
5. Run baseline replay validation against XRPL metadata.
6. Run same-fill AMM/CLOB allocation optimisation.
7. Summarise pair-level and transaction-level results.

See `docs/reproduction.md` for the command-level outline.

## Anonymity and Local Data

Do not commit local credentials, generated data, or machine-specific paths. The
`.gitignore` excludes common generated artifacts and local credential files.
