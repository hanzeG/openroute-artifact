# OpenRoute Artifact Commands

## 0. Environment

```bash
conda env create -f environment.yml
conda activate xrpl-amm-clob
pip install -e .
```

```bash
cd empirical/rust/tx_prebook_replay
cargo build --release
cd ../../..
```

## 1. Quick Check

```bash
PYTHONPATH=src pytest \
  tests/ci/test_smoke_imports.py \
  tests/rebuild/test_core_kernel.py \
  tests/rebuild/test_amm_rebuild.py
```

## 2. Configure a Run

```bash
PAIR=rlusd_xrp
BASE_LABEL=RLUSD
ISSUER=rMxCKbEDwqr76QuheSUMdEGf4B9xJ8m5De
CURRENCY_HEX=524C555344000000000000000000000000000000
LEDGER_START=100725835
LEDGER_END=101035981
WINDOW=ledger_${LEDGER_START}_${LEDGER_END}
RPC=https://s2.ripple.com:51234/
JOBS=8
SHARE_PROFILE=data/config.share
SHARE=ripple-ubri-share
SCHEMA=ripplex
TABLE_AMM=fact_amm_swaps
TABLE_CLOB=offers_fact_tx
TABLE_FEES=fact_amm_fees
```

```bash
mkdir -p artifacts/lists/${PAIR}/${WINDOW}
seq ${LEDGER_START} ${LEDGER_END} > artifacts/lists/${PAIR}/${WINDOW}/ledger_list.txt
LEDGER_LIST=artifacts/lists/${PAIR}/${WINDOW}/ledger_list.txt
```

```bash
cp data/config.share.example data/config.share
chmod 600 data/config.share
# Edit data/config.share before running Delta Sharing export commands.
cat configs/empirical/delta_sharing_tables.json
```

## 3. Fetch and Build Inputs

```bash
python empirical/scripts/empirical_fetch_full_ledger_metadata.py \
  --ledger-list ${LEDGER_LIST} \
  --outdir artifacts/metadata/${PAIR}/${WINDOW} \
  --rpc ${RPC} \
  --workers ${JOBS}
```

```bash
python empirical/scripts/empirical_export_window.py \
  --pair ${PAIR} \
  --share-profile ${SHARE_PROFILE} \
  --share ${SHARE} \
  --schema ${SCHEMA} \
  --table-amm ${TABLE_AMM} \
  --table-clob ${TABLE_CLOB} \
  --table-fees ${TABLE_FEES} \
  --ledger-start ${LEDGER_START} \
  --ledger-end ${LEDGER_END} \
  --base-currency ${CURRENCY_HEX} \
  --base-issuer ${ISSUER} \
  --counter-currency XRP \
  --counter-issuer "" \
  --output-dir artifacts/exports/${PAIR}/${WINDOW}
```

```bash
python empirical/scripts/empirical_build_full_tx_and_prebook_ledgers.py \
  --base-dir artifacts/exports/${PAIR} \
  --date-start 2025-12-08 \
  --date-end 2025-12-22 \
  --outdir artifacts/inputs/${PAIR}/${WINDOW}
```

```bash
python empirical/scripts/empirical_build_strict_direct_target_tx.py \
  --tx-sequence artifacts/inputs/${PAIR}/${WINDOW}/full_tx_sequence.parquet \
  --metadata-ndjson artifacts/metadata/${PAIR}/${WINDOW}/tx_metadata_full_merged.ndjson \
  --base-currency ${CURRENCY_HEX} \
  --base-label ${BASE_LABEL} \
  --counter-currency XRP \
  --counter-label XRP \
  --outdir artifacts/targets/${PAIR}/${WINDOW}/strict_direct
```

```bash
python empirical/scripts/empirical_fetch_prebook_from_ledger_data.py \
  --rpc ${RPC} \
  --ledger-list artifacts/inputs/${PAIR}/${WINDOW}/prebook_ledgers_full.txt \
  --outdir artifacts/prebook/${PAIR}/${WINDOW} \
  --issuer ${ISSUER} \
  --currency-hex ${CURRENCY_HEX} \
  --output-prefix book \
  --workers ${JOBS}
```

## 4. Replay Prebook

```bash
python empirical/scripts/empirical_replay_tx_prebook_rust.py \
  --ledger-start ${LEDGER_START} \
  --ledger-end ${LEDGER_END} \
  --book-gets-xrp artifacts/prebook/${PAIR}/${WINDOW}/book_getsXRP.ndjson \
  --book-gets-rusd artifacts/prebook/${PAIR}/${WINDOW}/book_getsrUSD.ndjson \
  --metadata-ndjson artifacts/metadata/${PAIR}/${WINDOW}/tx_metadata_full_merged.ndjson \
  --target-tx-file artifacts/targets/${PAIR}/${WINDOW}/strict_direct/required_tx.csv \
  --amm-swaps artifacts/exports/${PAIR}/${WINDOW}/amm_swaps \
  --clob-legs artifacts/exports/${PAIR}/${WINDOW}/clob_legs \
  --output-dir artifacts/replay/${PAIR}/${WINDOW}
```

## 5. Assemble Dataset Root

```bash
DATASET=artifacts/fit_inputs/${PAIR}/${WINDOW}/strict_direct_targets/two_week_isolated
mkdir -p ${DATASET}

cp artifacts/metadata/${PAIR}/${WINDOW}/tx_metadata_full_merged.ndjson \
  ${DATASET}/tx_metadata_full_merged.ndjson
cp artifacts/replay/${PAIR}/${WINDOW}/tx_prebook_replay_snapshots.ndjson \
  ${DATASET}/tx_prebook_snapshots.ndjson
cp artifacts/targets/${PAIR}/${WINDOW}/strict_direct/required_tx.csv \
  ${DATASET}/required_tx.csv
cp ${LEDGER_LIST} ${DATASET}/ledger_list.txt
cp -R artifacts/exports/${PAIR}/${WINDOW}/amm_swaps \
  ${DATASET}/amm_swaps_two_week.parquet
cp -R artifacts/exports/${PAIR}/${WINDOW}/amm_fees \
  ${DATASET}/amm_fees_two_week.parquet
```

```bash
cat > ${DATASET}/dataset_manifest.json <<EOF
{
  "pair": "${PAIR}",
  "window": "${WINDOW}",
  "base_currency": "${CURRENCY_HEX}",
  "base_label": "${BASE_LABEL}",
  "counter_currency": "XRP",
  "counter_label": "XRP"
}
EOF
```

If the target set includes `OfferCreate`, also place
`offer_tick_sizes_at_ledger_*.json` snapshots in `${DATASET}`.

## 6. Baseline Replay Check

```bash
python empirical/scripts/empirical_run_strict_direct_two_week_fit.py \
  --dataset-root ${DATASET} \
  --output-dir artifacts/compare/${PAIR}/${WINDOW}/isolated_fit \
  --aggregate-only \
  --top-relative-errors 50
```

## 7. Same-Fill Optimisation

```bash
python empirical/scripts/empirical_run_best_price_fixed_output_pairs.py \
  --pair ${PAIR}=${DATASET}/dataset_manifest.json \
  --output-root artifacts/optimisation/${WINDOW} \
  --pair-workers 1 \
  --jobs-per-pair ${JOBS} \
  --layer1-measurement-policy both
```

```bash
python empirical/scripts/empirical_summarize_layer1_same_fill_pairs.py \
  --batch-root artifacts/optimisation/${WINDOW} \
  --output-md artifacts/optimisation/${WINDOW}/summary.md \
  --output-json artifacts/optimisation/${WINDOW}/summary.json
```

Generated data stays under `artifacts/` and is ignored by Git.
