# OpenRoute Artifact

Commands below reproduce the empirical inputs, baseline replay checks, same-fill optimisation outputs, and derived result summaries.

## 0. Setup

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

## 1. Quick Check

```bash
PYTHONPATH=src pytest \
  tests/ci/test_smoke_imports.py \
  tests/ci/test_empirical_dry_run.py \
  tests/rebuild/test_core_kernel.py \
  tests/rebuild/test_amm_rebuild.py
```

## 2. Resolve the Ten Paper Pairs

```bash
python empirical/scripts/empirical_resolve_paper_pairs.py \
  --pair-config configs/empirical/paper_pairs.json \
  --share-profile data/config.share \
  --share ripple-ubri-share \
  --schema ripplex \
  --table-amm fact_amm_swaps \
  --output artifacts/config/paper_pairs.resolved.json
```

## 3. Build Dataset Roots

```bash
PAIR_LIST=(rlusd_xrp solo_xrp 666_xrp xah_xrp core_xrp fuzzy_xrp cny_xrp usdc_xrp 589_xrp mallard_xrp)
RPC=${XRPL_RIPPLE_S2_RPC}
JOBS=${XRPL_FETCH_WORKERS:-8}
```

Run the following loop. It creates one canonical dataset root per pair under `artifacts/fit_inputs`.

```bash
for PAIR in "${PAIR_LIST[@]}"; do
  eval "$(python empirical/scripts/empirical_pair_env.py \
    --pair-config artifacts/config/paper_pairs.resolved.json \
    --pair "${PAIR}")"

  EXPORT_DIR=artifacts/exports/${PAIR}/${WINDOW}
  INPUT_DIR=artifacts/inputs/${PAIR}/${WINDOW}
  METADATA_DIR=artifacts/metadata/${PAIR}/${WINDOW}
  TARGET_DIR=artifacts/targets/${PAIR}/${WINDOW}/strict_direct
  PREBOOK_DIR=artifacts/prebook/${PAIR}/${WINDOW}
  REPLAY_DIR=artifacts/replay/${PAIR}/${WINDOW}
  DATASET=artifacts/fit_inputs/${PAIR}/${WINDOW}/strict_direct_targets/two_week_isolated

  mkdir -p artifacts/lists/${PAIR}/${WINDOW}
  seq ${LEDGER_START} ${LEDGER_END} > artifacts/lists/${PAIR}/${WINDOW}/ledger_list.txt
  LEDGER_LIST=artifacts/lists/${PAIR}/${WINDOW}/ledger_list.txt

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

  python empirical/scripts/empirical_build_window_tx_and_prebook_ledgers.py \
    --export-dir ${EXPORT_DIR} \
    --outdir ${INPUT_DIR}

  python empirical/scripts/empirical_fetch_full_ledger_metadata.py \
    --ledger-list ${LEDGER_LIST} \
    --outdir ${METADATA_DIR} \
    --rpc ${RPC} \
    --workers ${JOBS}

  python empirical/scripts/empirical_build_strict_direct_target_tx.py \
    --tx-sequence ${INPUT_DIR}/full_tx_sequence.parquet \
    --metadata-ndjson ${METADATA_DIR}/tx_metadata_full_merged.ndjson \
    --base-currency ${CURRENCY_HEX} \
    --base-label ${BASE_LABEL} \
    --counter-currency XRP \
    --counter-label XRP \
    --outdir ${TARGET_DIR}

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

  python empirical/scripts/empirical_assemble_dataset_root.py \
    --pair-config artifacts/config/paper_pairs.resolved.json \
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

  python empirical/scripts/empirical_run_strict_direct_two_week_fit.py \
    --dataset-root ${DATASET} \
    --output-dir artifacts/compare/${PAIR}/${WINDOW}/final_fit \
    --aggregate-only \
    --write-non-strict-reports \
    --top-relative-errors 50
done
```

## 4. Same-Fill Optimisation

```bash
PAIR_ARGS=()
for PAIR in "${PAIR_LIST[@]}"; do
  eval "$(python empirical/scripts/empirical_pair_env.py \
    --pair-config artifacts/config/paper_pairs.resolved.json \
    --pair "${PAIR}")"
  DATASET=artifacts/fit_inputs/${PAIR}/${WINDOW}/strict_direct_targets/two_week_isolated
  PAIR_ARGS+=(--pair "${PAIR_LABEL}=${DATASET}/dataset_manifest.json")
done

python empirical/scripts/empirical_run_best_price_fixed_output_pairs.py \
  "${PAIR_ARGS[@]}" \
  --output-root artifacts/optimisation/${WINDOW} \
  --pair-workers 1 \
  --jobs-per-pair ${JOBS} \
  --layer1-measurement-policy both
```

## 5. Paper Data Summaries

```bash
python empirical/scripts/empirical_summarize_pair_setup.py \
  --pair-config artifacts/config/paper_pairs.resolved.json \
  --output-dir artifacts/results/pair_setup

python empirical/scripts/empirical_summarize_baseline_fit_pairs.py \
  --pair-config artifacts/config/paper_pairs.resolved.json \
  --compare-root artifacts/compare \
  --output-dir artifacts/results/baseline_fit

python empirical/scripts/empirical_summarize_layer1_same_fill_pairs.py \
  --batch-root artifacts/optimisation/${WINDOW} \
  --output-md artifacts/results/same_fill_summary.md \
  --output-json artifacts/results/same_fill_summary.json

python empirical/scripts/empirical_summarize_paper_data.py \
  --results-root artifacts/optimisation/${WINDOW} \
  --pair-config artifacts/config/paper_pairs.resolved.json \
  --fit-input-root artifacts/fit_inputs \
  --output-dir artifacts/results/paper_data

python empirical/scripts/empirical_extract_same_fill_cases.py \
  --results-root artifacts/optimisation/${WINDOW} \
  --tx-hash 60B4071AB9FFDF43C4190DE1C216FD57E366A23859FAFE3DB1364C209CABCC80 \
  --tx-hash 21D4A30EEF00C483C53171E96FB66591DC82FF480CFF6A3A48ACA506E0B1DF77 \
  --tx-hash 3B56837BD05F0DC6B25D660423B444703D89FD1BE1F191C8E1BB83AA8EC6C6E3 \
  --output-json artifacts/results/case_studies/same_fill_cases.json
```

Baseline replay reports are written to `artifacts/compare/*/*/final_fit/error_analysis.md`.
