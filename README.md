# OpenRoute Artifact

Run the commands below in order to reproduce the empirical inputs, baseline replay checks, same-fill optimisation outputs, and paper-data summaries.

## 0. Clone

```bash
git clone <anonymous-artifact-repository-url>
cd openroute-artifact
```

## 1. Environment Setup

```bash
conda env create -f environment.yml
conda activate xrpl-amm-clob
pip install -e .

cd empirical/rust/tx_prebook_replay
cargo build --release
cd ../../..
```

```bash
cp .env.example .env
set -a
source .env
set +a
```

Copy the Delta Sharing profile into `data/config.share`:

```bash
cp data/config.share.example data/config.share
chmod 600 data/config.share
```

The repository includes the lightweight Delta-derived metadata used to define
the experiment: `data/token_selection/target_window_rankings/` contains the
pair-selection ranking tables, and
`configs/empirical/paper_pairs.resolved.json` contains the resolved
currency/issuer identifiers for the ten paper pairs. A Delta Sharing credential
is only needed if you want to regenerate these files from the Ripple UBRI Delta
Sharing tables or rerun the raw AMM/CLOB/fee row export in Section 4.1. The raw
Delta Sharing tables and credential profile are not included in this repository.

## 2. Quick Check

```bash
PYTHONPATH=src pytest tests
```

## 3. Pair Selection Inputs

The ten paper pairs are already resolved in
`configs/empirical/paper_pairs.resolved.json`. The source ranking tables used to
select them are included under `data/token_selection/target_window_rankings/`.

To regenerate the resolved pair file from Delta Sharing, run:

```bash
python empirical/scripts/empirical_resolve_paper_pairs.py \
  --pair-config configs/empirical/paper_pairs.json \
  --share-profile data/config.share \
  --share ripple-ubri-share \
  --schema ripplex \
  --table-amm fact_amm_swaps \
  --output configs/empirical/paper_pairs.resolved.json
```

Expected output:

```text
configs/empirical/paper_pairs.resolved.json
```

## 4. Build Dataset Roots

The commands in this section create one canonical dataset root per pair under `artifacts/fit_inputs`.

### 4.0 Shared Shell Context

Run this once per shell session before the remaining commands.

```bash
PAIR_LIST=(rlusd_xrp solo_xrp 666_xrp xah_xrp core_xrp fuzzy_xrp cny_xrp usdc_xrp 589_xrp mallard_xrp)
RPC=${XRPL_RIPPLE_S2_RPC}
JOBS=${XRPL_FETCH_WORKERS:-8}

load_pair() {
  PAIR="$1"
  eval "$(python empirical/scripts/empirical_pair_env.py \
    --pair-config configs/empirical/paper_pairs.resolved.json \
    --pair "${PAIR}")"

  export EXPORT_DIR=artifacts/exports/${PAIR}/${WINDOW}
  export INPUT_DIR=artifacts/inputs/${PAIR}/${WINDOW}
  export METADATA_DIR=artifacts/metadata/${PAIR}/${WINDOW}
  export TARGET_DIR=artifacts/targets/${PAIR}/${WINDOW}/strict_direct
  export PREBOOK_DIR=artifacts/prebook/${PAIR}/${WINDOW}
  export REPLAY_DIR=artifacts/replay/${PAIR}/${WINDOW}
  export DATASET=artifacts/fit_inputs/${PAIR}/${WINDOW}/strict_direct_targets/two_week_isolated
  export LEDGER_LIST=artifacts/lists/${PAIR}/${WINDOW}/ledger_list.txt

  mkdir -p artifacts/lists/${PAIR}/${WINDOW}
  seq ${LEDGER_START} ${LEDGER_END} > ${LEDGER_LIST}
}
```

### 4.1 Export Delta Sharing Rows

Exports AMM swaps, CLOB legs, and AMM fees for each pair/window.

```bash
for PAIR in "${PAIR_LIST[@]}"; do
  load_pair "${PAIR}"

  python empirical/scripts/empirical_export_window.py \
    --pair ${PAIR} \
    --share-profile data/config.share \
    --share ripple-ubri-share \
    --schema ripplex \
    --table-amm fact_amm_swaps \
    --table-clob offers_fact_tx \
    --table-fees fact_amm_fees \
    --ledger-start ${LEDGER_START} \
    --ledger-end ${LEDGER_END} \
    --base-currency ${CURRENCY_HEX} \
    --base-issuer ${ISSUER} \
    --counter-currency XRP \
    --counter-issuer "" \
    --output-dir ${EXPORT_DIR}
done
```

Expected outputs per pair:

```text
artifacts/exports/<pair>/<window>/amm_swaps
artifacts/exports/<pair>/<window>/clob_legs
artifacts/exports/<pair>/<window>/amm_fees
```

### 4.2 Build Transaction Lists and Fetch Metadata

Builds the pair-level transaction sequence and fetches full ledger metadata for the window.

```bash
for PAIR in "${PAIR_LIST[@]}"; do
  load_pair "${PAIR}"

  python empirical/scripts/empirical_build_window_tx_and_prebook_ledgers.py \
    --export-dir ${EXPORT_DIR} \
    --outdir ${INPUT_DIR}

  python empirical/scripts/empirical_fetch_full_ledger_metadata.py \
    --ledger-list ${LEDGER_LIST} \
    --outdir ${METADATA_DIR} \
    --rpc ${RPC} \
    --workers ${JOBS}
done
```

Expected outputs per pair:

```text
artifacts/inputs/<pair>/<window>/full_tx_sequence.parquet
artifacts/inputs/<pair>/<window>/prebook_ledgers_full.txt
artifacts/metadata/<pair>/<window>/tx_metadata_full_merged.ndjson
```

### 4.3 Select Direct Target Transactions

Filters the exported pair activity to direct single-path target transactions.

```bash
for PAIR in "${PAIR_LIST[@]}"; do
  load_pair "${PAIR}"

  python empirical/scripts/empirical_build_strict_direct_target_tx.py \
    --tx-sequence ${INPUT_DIR}/full_tx_sequence.parquet \
    --metadata-ndjson ${METADATA_DIR}/tx_metadata_full_merged.ndjson \
    --base-currency ${CURRENCY_HEX} \
    --base-label ${BASE_LABEL} \
    --counter-currency XRP \
    --counter-label XRP \
    --outdir ${TARGET_DIR}
done
```

Expected output per pair:

```text
artifacts/targets/<pair>/<window>/strict_direct/required_tx.csv
```

### 4.4 Fetch Prebook and Account-Line Snapshots

Fetches ledger-level book offers and account-line snapshots required for tx-level pre-state replay.

```bash
for PAIR in "${PAIR_LIST[@]}"; do
  load_pair "${PAIR}"

  python empirical/scripts/empirical_fetch_prebook_from_ledger_data.py \
    --rpc ${RPC} \
    --ledger-list ${INPUT_DIR}/prebook_ledgers_full.txt \
    --outdir ${PREBOOK_DIR} \
    --issuer ${ISSUER} \
    --currency-hex ${CURRENCY_HEX} \
    --output-prefix book \
    --workers ${JOBS}

  python empirical/scripts/empirical_prepare_full_replay_account_lines_inputs.py \
    --metadata-ndjson ${METADATA_DIR}/tx_metadata_full_merged.ndjson \
    --required-tx-csv ${TARGET_DIR}/required_tx.csv \
    --output-csv artifacts/account_lines/${PAIR}/${WINDOW}/targets.csv

  python empirical/scripts/empirical_fetch_account_lines_snapshots.py \
    --input-csv artifacts/account_lines/${PAIR}/${WINDOW}/targets.csv \
    --outdir artifacts/account_lines/${PAIR}/${WINDOW} \
    --rpc ${RPC} \
    --workers ${JOBS} \
    --rps ${JOBS}
done
```

Expected outputs per pair:

```text
artifacts/prebook/<pair>/<window>/book_getsXRP.ndjson
artifacts/prebook/<pair>/<window>/book_getsrUSD.ndjson
artifacts/account_lines/<pair>/<window>/account_lines_snapshots.ndjson
```

The `book_getsrUSD.ndjson` name is a legacy token-side filename; it is used for all issued-token/XRP pairs.

### 4.5 Replay Tx-Level Prebook State

Replays all preceding transactions in the window to produce the prebook snapshot seen by each target transaction.

```bash
for PAIR in "${PAIR_LIST[@]}"; do
  load_pair "${PAIR}"

  python empirical/scripts/empirical_replay_tx_prebook_rust.py \
    --ledger-start ${LEDGER_START} \
    --ledger-end ${LEDGER_END} \
    --book-gets-xrp ${PREBOOK_DIR}/book_getsXRP.ndjson \
    --book-gets-rusd ${PREBOOK_DIR}/book_getsrUSD.ndjson \
    --metadata-ndjson ${METADATA_DIR}/tx_metadata_full_merged.ndjson \
    --target-tx-file ${TARGET_DIR}/required_tx.csv \
    --account-lines-snapshots artifacts/account_lines/${PAIR}/${WINDOW}/account_lines_snapshots.ndjson \
    --amm-swaps ${EXPORT_DIR}/amm_swaps \
    --clob-legs ${EXPORT_DIR}/clob_legs \
    --output-dir ${REPLAY_DIR}
done
```

Expected output per pair:

```text
artifacts/replay/<pair>/<window>/tx_prebook_replay_snapshots.ndjson
```

### 4.6 Assemble Dataset Roots and Enrich Ledger Fields

Assembles the canonical dataset root and fetches transfer-rate and tick-size inputs used by the replay model.

```bash
for PAIR in "${PAIR_LIST[@]}"; do
  load_pair "${PAIR}"

  python empirical/scripts/empirical_assemble_dataset_root.py \
    --pair-config configs/empirical/paper_pairs.resolved.json \
    --pair ${PAIR} \
    --metadata-ndjson ${METADATA_DIR}/tx_metadata_full_merged.ndjson \
    --tx-prebook-snapshots ${REPLAY_DIR}/tx_prebook_replay_snapshots.ndjson \
    --required-tx ${TARGET_DIR}/required_tx.csv \
    --ledger-list ${LEDGER_LIST} \
    --amm-swaps ${EXPORT_DIR}/amm_swaps \
    --amm-fees ${EXPORT_DIR}/amm_fees \
    --output-dir ${DATASET}

  python empirical/scripts/empirical_prepare_issuer_transfer_rate_ranges.py \
    --dataset-root ${DATASET} \
    --rpc-url ${RPC} \
    --write

  python empirical/scripts/empirical_fetch_offer_tick_sizes.py \
    --required-tx ${DATASET}/required_tx.csv \
    --metadata-ndjson ${DATASET}/tx_metadata_full_merged.ndjson \
    --outdir ${DATASET} \
    --rpc ${RPC} \
    --workers ${JOBS}
done
```

Expected output per pair:

```text
artifacts/fit_inputs/<pair>/<window>/strict_direct_targets/two_week_isolated/dataset_manifest.json
```

### 4.7 Preliminary Replay Fit and Account-Offers Snapshots

The preliminary fit identifies target transactions that need account-offers snapshots for fundedness reconstruction.

```bash
for PAIR in "${PAIR_LIST[@]}"; do
  load_pair "${PAIR}"

  python empirical/scripts/empirical_run_strict_direct_two_week_fit.py \
    --dataset-root ${DATASET} \
    --output-dir artifacts/compare/${PAIR}/${WINDOW}/preliminary_fit \
    --aggregate-only \
    --write-structure-false-reports \
    --top-relative-errors 50

  python empirical/scripts/empirical_prepare_account_offers_snapshot_inputs.py \
    --report-dir artifacts/compare/${PAIR}/${WINDOW}/preliminary_fit/reports \
    --tx-prebook-snapshots ${DATASET}/tx_prebook_snapshots.ndjson \
    --outdir artifacts/account_offers/${PAIR}/${WINDOW} \
    --allow-empty

  if [ "$(wc -l < artifacts/account_offers/${PAIR}/${WINDOW}/account_offers_targets.csv)" -gt 1 ]; then
    python empirical/scripts/empirical_fetch_account_offers_snapshots.py \
      --input-csv artifacts/account_offers/${PAIR}/${WINDOW}/account_offers_targets.csv \
      --outdir artifacts/account_offers/${PAIR}/${WINDOW} \
      --rpc ${RPC} \
      --workers ${JOBS} \
      --rps ${JOBS}

    cp artifacts/account_offers/${PAIR}/${WINDOW}/account_offers_ok.ndjson \
      ${DATASET}/account_offers_snapshots.ndjson
  fi
done
```

Expected outputs per pair:

```text
artifacts/compare/<pair>/<window>/preliminary_fit/aggregate.json
artifacts/account_offers/<pair>/<window>/account_offers_targets.csv
```

### 4.8 Final Baseline Replay Fit

Runs the final baseline replay check against the completed dataset root.

```bash
for PAIR in "${PAIR_LIST[@]}"; do
  load_pair "${PAIR}"

  python empirical/scripts/empirical_run_strict_direct_two_week_fit.py \
    --dataset-root ${DATASET} \
    --output-dir artifacts/compare/${PAIR}/${WINDOW}/final_fit \
    --aggregate-only \
    --write-non-strict-reports \
    --top-relative-errors 50
done
```

Expected outputs per pair:

```text
artifacts/compare/<pair>/<window>/final_fit/aggregate.json
artifacts/compare/<pair>/<window>/final_fit/error_analysis.md
```

## 5. Same-Fill Optimisation

Build the optimiser pair arguments from the dataset manifests:

```bash
PAIR_ARGS=()
for PAIR in "${PAIR_LIST[@]}"; do
  load_pair "${PAIR}"
  PAIR_ARGS+=(--pair "${PAIR_LABEL}=${DATASET}/dataset_manifest.json")
done
```

Run the same-fill optimisation for all pairs:

```bash
python empirical/scripts/empirical_run_best_price_fixed_output_pairs.py \
  "${PAIR_ARGS[@]}" \
  --output-root artifacts/optimisation/${WINDOW} \
  --pair-workers 1 \
  --jobs-per-pair ${JOBS} \
  --layer1-measurement-policy both
```

Expected output:

```text
artifacts/optimisation/<window>/<PAIR_LABEL>/results.ndjson
```

## 6. Paper Data Summaries

These commands produce compact JSON/Markdown summaries used to check the paper tables, figures, and case studies.

### 6.1 Pair Setup Summary

```bash
python empirical/scripts/empirical_summarize_pair_setup.py \
  --pair-config configs/empirical/paper_pairs.resolved.json \
  --output-dir artifacts/results/pair_setup
```

### 6.2 Baseline Fit Summary

```bash
python empirical/scripts/empirical_summarize_baseline_fit_pairs.py \
  --pair-config configs/empirical/paper_pairs.resolved.json \
  --compare-root artifacts/compare \
  --output-dir artifacts/results/baseline_fit
```

### 6.3 Same-Fill Optimisation Summary

```bash
python empirical/scripts/empirical_summarize_layer1_same_fill_pairs.py \
  --batch-root artifacts/optimisation/${WINDOW} \
  --output-md artifacts/results/same_fill_summary.md \
  --output-json artifacts/results/same_fill_summary.json
```

### 6.4 Paper Analysis Summary

```bash
python empirical/scripts/empirical_summarize_paper_data.py \
  --results-root artifacts/optimisation/${WINDOW} \
  --pair-config configs/empirical/paper_pairs.resolved.json \
  --fit-input-root artifacts/fit_inputs \
  --output-dir artifacts/results/paper_data
```

### 6.5 Case-Study Rows

```bash
python empirical/scripts/empirical_extract_same_fill_cases.py \
  --results-root artifacts/optimisation/${WINDOW} \
  --tx-hash 60B4071AB9FFDF43C4190DE1C216FD57E366A23859FAFE3DB1364C209CABCC80 \
  --tx-hash 21D4A30EEF00C483C53171E96FB66591DC82FF480CFF6A3A48ACA506E0B1DF77 \
  --tx-hash 3B56837BD05F0DC6B25D660423B444703D89FD1BE1F191C8E1BB83AA8EC6C6E3 \
  --output-json artifacts/results/case_studies/same_fill_cases.json
```

Expected summary outputs:

```text
artifacts/results/pair_setup/pair_setup_summary.json
artifacts/results/baseline_fit/baseline_fit_summary.json
artifacts/results/same_fill_summary.json
artifacts/results/paper_data/paper_data_summary.json
artifacts/results/case_studies/same_fill_cases.json
```
