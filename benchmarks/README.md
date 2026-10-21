# Benchmarks

`bench_inference.py` times exact inference on the pgmpy-backed BN (ADR-0001 §9 Q2/Q7; target ≤ 30 s, stretch ≤ 5 s).

```bash
# real fitted model (path must be under the current directory)
python benchmarks/bench_inference.py --model out/bayesian_network/model_fit/bayesian_network.pkl --output report.json
# synthetic
python benchmarks/bench_inference.py --nodes 60 --card 4 --max-parents 4 --rows 50000
```

The JSON report records CPU model / cores / RAM / library versions (no hostnames) so runs on different machines are comparable. Reports are local files: do not commit reports or models from internal data.

## Reference numbers (2026-10-20, Apple M-series laptop, 10 cores, 32 GB, Python 3.10, pgmpy 1.1.2)

| Case | Slowest median query |
|---|---|
| Students example BN (46 nodes, 135 edges, fitted by `infer-bn`) | 17 ms |
| Synthetic 30 nodes, card 4, ≤ 3 parents | 7 ms |
| Synthetic 60 nodes, card 4, ≤ 4 parents | 0.23 s |
| Synthetic 60 nodes, card 6, ≤ 5 parents, 100k rows | did not finish in > 10 min |

Takeaways: restricting `query(nodes=[...])` is what makes typical use cheap (a full all-node query is ~1000× slower than a targeted one). Exact variable elimination is still exponential in treewidth, so very dense, high-cardinality networks remain out of reach — reduce discretisation buckets / parents (`max_categories`, `proportion_threshold`) before reaching for a different engine.

The ADR's "7 minute" case is **not** reproduced by any of these, and it has **not been verified on production data**: that check is optional and internal. Anyone with access to such a model can run the harness locally (see the legacy-export section below); the result need not be added to this repo.

## Benchmarking a model trained with causal-canvas < 0.2.0 (no retraining, no data)

Old pickles do not load in 0.2.0. `export_legacy_model.py` runs in the **old** environment (causalnex installed), reads the pickle and writes a JSON of structure + probability tables only; the benchmark rebuilds the network from it:

```bash
# old environment
python benchmarks/export_legacy_model.py --model path/to/bayesian_network.pkl --output model_tables.json
# new environment
python benchmarks/bench_inference.py --tables model_tables.json --output report.json
```

Fidelity check (students example, 45 nodes / 157 edges): marginals from the rebuilt network match the old causalnex `InferenceEngine` to 7e-15. Local legacy models (8–45 nodes) all run in ≤ 18 ms; none of them is the ADR's 7-minute production model.
