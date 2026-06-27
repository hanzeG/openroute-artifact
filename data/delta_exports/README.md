This directory contains the bundled Delta Sharing query results used by the
paper. These parquet files replace the credentialed Delta Sharing export step in
the default reproduction path.

Layout:

```text
data/delta_exports/<pair>/<window>/amm_swaps/
data/delta_exports/<pair>/<window>/clob_legs/
data/delta_exports/shared/<window>/amm_fees/
```

AMM swaps and CLOB legs are pair-specific. CLOB rows include the `ledger_index`
and `transaction_index` ordering columns required by the reproduction pipeline.
AMM fee rows are shared across pairs for the ledger window because they are
queried by ledger range rather than by token pair.
