# Empirical Scripts

This directory contains the data-processing scripts used by the OpenRoute
artifact. The scripts are organised around four stages:

1. Fetch XRPL metadata and ledger-state inputs.
2. Build direct XRP-token targets and transaction-level pre-state inputs.
3. Replay prebooks and validate the baseline execution model.
4. Run same-fill AMM/CLOB allocation optimisation and summarise results.

The full workflow is documented in `../docs/reproduction.md`. Large generated
outputs should be written to ignored local directories such as `artifacts/` or
`.tmp/`.

Credentials are local-only. Start from `../data/config.share.example`, write the
real credential file as `data/config.share`, and keep it out of Git.
