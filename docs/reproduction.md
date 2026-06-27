# Reproduction Outline

This document records the lightweight reproduction path for regenerating the
OpenRoute experiment data from XRPL inputs. The commands below are templates:
replace ledger ranges, pair names, and input/output paths with the desired run.

## 1. Configure Access

Create a local credential file from the template:

```bash
cp data/config.share.example data/config.share
chmod 600 data/config.share
```

The credential file is local-only and ignored by Git.

## 2. Fetch Ledger and Transaction Inputs

Fetch validated transaction metadata and the ledger-state inputs needed for
pre-state construction:

```bash
python empirical/scripts/empirical_fetch_full_ledger_metadata.py \
  --ledger-start <ledger_start> \
  --ledger-end <ledger_end> \
  --output-dir artifacts/metadata/<pair>/<window>
```

Fetch or rebuild CLOB/AMM ledger inputs:

```bash
python empirical/scripts/empirical_fetch_prebook_from_ledger_data.py \
  --ledger-start <ledger_start> \
  --ledger-end <ledger_end> \
  --output-dir artifacts/prebook/<pair>/<window>
```

For maker-fundedness-sensitive runs, prepare and fetch account-offer and
account-line snapshots:

```bash
python empirical/scripts/empirical_prepare_account_offers_snapshot_inputs.py \
  --targets <targets.ndjson> \
  --output <account_offer_inputs.ndjson>

python empirical/scripts/empirical_fetch_account_offers_snapshots.py \
  --inputs <account_offer_inputs.ndjson> \
  --output <account_offers_ok.ndjson>
```

## 3. Build Target and Pre-State Inputs

Construct the strict direct XRP-token target set:

```bash
python empirical/scripts/empirical_build_strict_direct_target_tx.py \
  --metadata <metadata.ndjson> \
  --pair <pair> \
  --output <targets.ndjson>
```

Build transaction-prebook replay inputs:

```bash
python empirical/scripts/empirical_build_full_tx_and_prebook_ledgers.py \
  --targets <targets.ndjson> \
  --metadata <metadata.ndjson> \
  --output-dir <replay_input_dir>
```

## 4. Replay Transaction Prebooks

Build and run the Rust replay helper:

```bash
cd empirical/rust/tx_prebook_replay
cargo build --release
cd ../../..

python empirical/scripts/empirical_replay_tx_prebook_rust.py \
  --input-dir <replay_input_dir> \
  --output <tx_prebook_snapshots.ndjson>
```

## 5. Validate Baseline Replay

Run the isolated replay/fit comparison against XRPL metadata:

```bash
python empirical/scripts/empirical_run_strict_direct_two_week_fit.py \
  --targets <targets.ndjson> \
  --metadata <metadata.ndjson> \
  --tx-prebook-snapshots <tx_prebook_snapshots.ndjson> \
  --output-dir <baseline_fit_dir>
```

The key baseline outputs are `aggregate.json`, `summary.json`, and per-transaction
reports under the output directory.

## 6. Run Same-Fill Optimisation

Run pair-level same-fill optimisation on fully fitted baseline transactions:

```bash
python empirical/scripts/empirical_run_best_price_fixed_output_pairs.py \
  --fit-root <baseline_fit_root> \
  --output-root <optimisation_output_root> \
  --jobs <n>
```

Summarise the outputs:

```bash
python empirical/scripts/empirical_summarize_layer1_same_fill_pairs.py \
  --input-root <optimisation_output_root> \
  --output <summary.json>
```

## 7. Practical Notes

- Full two-week, multi-pair runs are storage- and time-intensive.
- Generated metadata, replay snapshots, and pair-level outputs should stay under
  `artifacts/`, `.tmp/`, or another ignored local path.
- The scripts operate in XRPL amount domains; do not post-process decimal display
  strings as a substitute for replay outputs.
