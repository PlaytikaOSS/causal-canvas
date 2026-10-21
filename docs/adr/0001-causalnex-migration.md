# ADR-0001: Migration off `causalnex` for `causal-canvas` 0.2.0

## 1. Decision

**(d) Hybrid: replace the BN / inference layer with `pgmpy` 1.x, keep the vendored NOTEARS-MLP under explicit ownership, and adopt `DoWhy` 0.14 as a separate complement for the `infer_do_why.py` causal-effect path.**

## 2. Status

**ACCEPTED** — proposed 2026-04-25 in internal PR #56; accepted and implemented 2026-10-20 (see §11). Real-data benchmarks (§9 Q2) remain open.

## 3. Context

- `causalnex` 0.12.1 is functionally abandoned: last release 2023-06-22, last commit on `develop` 2024-02-10 (a dependency-bump merge), **0 commits in the last 90 days**, **0 distinct contributors in the last 12 months**, and issue [#242 "Is this project being maintained?"](https://github.com/mckinsey/causalnex/issues/242) (opened 2025-09-26) has had zero maintainer response in seven months.
- The transitive pin chain (`torch==2.2.1`, `pyarrow==15.0.0`) is exposed to **6 distinct CVEs**, headlined by [GHSA-53q9-r3pm-6pq6 / CVE-2025-32434](https://github.com/advisories/GHSA-53q9-r3pm-6pq6) (Critical 9.3, RCE via `torch.load(weights_only=True)`) and [GHSA-63cw-57p8-fm3p / CVE-2026-24747](https://github.com/advisories/GHSA-63cw-57p8-fm3p) (High 8.8, RCE via crafted checkpoint). These cannot be patched while `causalnex` stays in the tree.
- Inference performance pain (~7 min for some queries on modest data) is inherent to causalnex's pure-Python `pgmpy<0.1.20`-era engine. Replacement should be no slower; mature engines with C++ kernels (pyAgrum) or current-pgmpy are realistic improvements.
- `causal-canvas` is at v0.1.0 and the next release is intentionally a breaking 0.2.0 — we have a free hand on API breaks. Pickle preservation is not a hard requirement.

## 4. Considered options — scorecard

All facts below are primary-sourced (GitHub/GitLab/PyPI), verified 2026-04-25.

| Candidate | Latest release | 90d commits | 12mo contribs | License | py.typed | NOTEARS | BN + CPT fit | do() / counterfactual | Bus factor (top author L12M) | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| **pgmpy 1.1.0** | [2026-04-01](https://pypi.org/project/pgmpy/1.1.0/) | 256 | 101 | MIT | No | MISSING | FULL (`DiscreteBayesianNetwork`, `DAG.fit`) | FULL (`CausalInference.query(do=...)`, `estimate_ate`); counterfactuals PARTIAL ([CausalInference.py L687, L920](https://github.com/pgmpy/pgmpy/blob/dev/pgmpy/inference/CausalInference.py)) | 31.5% (`ankurankan`) | **WINNER** |
| pyAgrum 2.3.2 | [2026-01-26](https://pypi.org/project/pyAgrum/2.3.2/) | 17 | 11 | **Dual LGPL-3.0-or-later OR MIT** ([LICENSE.md](https://gitlab.com/agrumery/aGrUM/-/raw/master/LICENSE.md)) | No | MISSING | FULL ([BNLearning](https://pyagrum.readthedocs.io/en/latest/BNLearning.html), [BNInference](https://pyagrum.readthedocs.io/en/latest/BNInference.html)) | FULL interventions (`causalImpact`, `doCalculusWithObservation`, Shpitser-Pearl `identifyingIntervention`); counterfactuals **MISSING** | **62.9%** (Wuilemin/Wuillemin combined) | Strong technically but high single-PI risk + dual-license requires legal MIT election; **runner-up** |
| bnlearn 0.13.0 (Taskesen) | [2026-03-07](https://pypi.org/project/bnlearn/0.13.0/) | 11 | 3 (1 human) | MIT | No | MISSING | PARTIAL (wraps pgmpy; **no `do=` parameter**, see open [#79](https://github.com/erdogant/bnlearn/issues/79) since 2023-05) | MISSING | **92.0%** (`erdogant`) | Solo maintainer + no do() — strictly inferior to using pgmpy directly |
| pomegranate 1.1.2 | [2025-02-07](https://pypi.org/project/pomegranate/1.1.2/) (>14 mo stale) | **0** | **0** | MIT | No | MISSING (README: "BN structure learning … is still incomplete") | PARTIAL (no do-intervention) | MISSING | ~100% (dormant) | DORMANT post-rewrite, 35 open issues >6 mo old |
| pyro 1.9.1 | [2024-06-02](https://pypi.org/project/pyro-ppl/1.9.1/) (~22 mo stale) | **0** | **1** | Apache-2.0 | Yes | N/A (PPL) | Non-idiomatic for tabular CPTs | PARTIAL (do via ChiRho extension) | 1 | Stale; wrong tool shape |
| numpyro 0.20.1 | [2026-03-25](https://pypi.org/project/numpyro/0.20.1/) | 29 | 21 | Apache-2.0 | No | N/A (PPL) | Non-idiomatic for tabular CPTs | PARTIAL (manual rewrite) | ~3 | Wrong tool shape for discrete-CPT BN |
| **DoWhy 0.14** (orthogonal) | [2025-11-08](https://github.com/py-why/dowhy/releases/tag/v0.14) | 48 (incl. ~33 bot) | 17 | MIT (PyWhy org; Microsoft + AWS contributing — [pywhy.org](https://www.pywhy.org/)) | No | MISSING (delegates to `causal-learn`) | N/A — **NOT a BN library, does not fit CPTs** | FULL do-calculus (`identify_effect → estimate_effect → refute_estimate`); counterfactuals via `gcm.counterfactual_samples` | ~2 humans | Adopt as **complement** for `infer_do_why.py`, NOT as BN substitute |
| Soft-fork causalnex (option b) | n/a | n/a | Playtika-internal | Apache-2.0 (upstream) | maintain ourselves | keep | keep | keep | n/a | Inherits API surface + obligation to upgrade torch/pyarrow ourselves; ongoing cost |

Two facts that change the search:

- **NOTEARS is missing from every candidate**, so the algorithm itself must be vendored regardless of which BN library wins. Our existing [`causal_canvas/custom_notears_mlp.py`](../../causal_canvas/custom_notears_mlp.py) already encodes Playtika-specific changes; the upstream reference [`xunzheng/notears`](https://github.com/xunzheng/notears) is itself dormant (last commit 2022-12-09). Owning NOTEARS-MLP outright is therefore a near-zero marginal increase over what we already do.
- **`bnlearn` is a thin wrapper over `pgmpy`** with a 92% solo-author bus factor and no `do=` API. There is no scenario where `bnlearn` beats `pgmpy` directly.

## 5. Decision drivers — non-negotiables vs. winner evidence

| Criterion | pgmpy 1.1.0 evidence |
|---|---|
| Production-grade, not academic-only | Released 1.1.0 less than four weeks ago; 1.0.0 stabilization in 2025-03 with explicit breaking-change documentation in CHANGELOG. |
| Not solo-maintainer | 101 distinct contributors L12M; top author 31.5% (healthy). |
| Not declining commit activity | 256 commits L90D — highest of any candidate. |
| Types/stubs and Python ≥3.10 | py.typed missing → mitigation: ship local stubs in `causal-canvas/typings/`; Python ≥3.10 supported. |
| Permissive license | MIT, verified in [LICENSE](https://github.com/pgmpy/pgmpy/blob/dev/LICENSE) and PyPI metadata. |
| Covers our surface | BN+CPT fit ✓, marginal/conditional ✓, do-intervention ✓ ([CausalInference.py L920](https://github.com/pgmpy/pgmpy/blob/dev/pgmpy/inference/CausalInference.py)). NOTEARS gap is closed by keeping our vendored MLP. |

pyAgrum loses the tiebreaker on three points: (a) dual LGPL/MIT requires explicit legal MIT election before adoption, (b) 62.9% single-PI bus factor is the same failure mode that produced the causalnex situation, (c) no public production users found from primary sources — a recurrence-risk signal we should not ignore twice.

## 6. Numerical-parity gate — 5-node toy network (paper spec for Phase B)

**Network**: A → B, A → C, B → D, C → D, D → E. All nodes binary {0,1}.

**CPTs** (hand-specified, deterministic):

- P(A=1) = 0.4
- P(B=1|A=0) = 0.2; P(B=1|A=1) = 0.7
- P(C=1|A=0) = 0.3; P(C=1|A=1) = 0.8
- P(D=1|B,C) = [0.05, 0.4, 0.5, 0.95] for (B,C) ∈ {(0,0),(0,1),(1,0),(1,1)}
- P(E=1|D=0) = 0.1; P(E=1|D=1) = 0.85

**Reference queries** (computed by hand / reference implementation, verified independently):

1. Marginal **P(D=1)** ≈ 0.4534
2. Conditional **P(B=1 | D=1)** ≈ 0.7395
3. Interventional **P(D=1 | do(B=1))** ≈ 0.6770

(Phase B will compute exact reference values using a third reference — e.g. NetworkX hand-calc or pyAgrum LazyPropagation — and pin them as test fixtures.)

**Acceptance**: agreement within 1e-6 absolute on all three queries. Discrete exact inference is deterministic — any mismatch is an API-mapping bug.

**API mapping — pgmpy 1.1**:

```python
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.inference.CausalInference import CausalInference

bn = DiscreteBayesianNetwork([('A','B'),('A','C'),('B','D'),('C','D'),('D','E')])
bn.add_cpds(...)                                # TabularCPDs as above
ve = VariableElimination(bn)
ve.query(['D']).values                          # marginal
ve.query(['B'], evidence={'D': 1}).values       # conditional
ci = CausalInference(bn)
ci.query(variables=['D'], do={'B': 1}).values   # do-intervention
```

**API mapping — pyAgrum 2.3 (runner-up, kept for parity cross-check)**:

```python
import pyagrum as gum
import pyagrum.causal as csl

bn = gum.fastBN("A->B;A->C;B->D;C->D;D->E")     # then overwrite CPTs explicitly
ie = gum.LazyPropagation(bn); ie.makeInference()
ie.posterior('D')                                # marginal
ie.setEvidence({'D': 1}); ie.makeInference(); ie.posterior('B')  # conditional
cm = csl.CausalModel(bn)
_, p, _ = csl.causalImpact(cm, on='D', doing='B', values={'B': 1})  # do-intervention
```

Hard-fail clauses:

- pyAgrum **cannot** express twin-network counterfactuals natively. We do not test counterfactuals in this gate (out of scope for the BN replacement; counterfactuals move to DoWhy in `infer_do_why.py`).
- bnlearn fails the gate by construction — no `do=` API, so query (3) is unreachable. This is one reason it is excluded.

## 7. Migration cost estimate (engineer-days)

**Option (d) — recommended**:

| Workstream | Days | Notes |
|---|---|---|
| `nonlinear_notears.py` cleanup; drop causalnex deps from vendored MLP | 2 | `custom_notears_mlp.py` already owns the modifications |
| Replace `causalnex.network.BayesianNetwork` → `pgmpy.DiscreteBayesianNetwork` (+ structure adapter) | 3 | API mapping; pgmpy 1.0 renamed `BayesianNetwork → DiscreteBayesianNetwork`, `ci_pillai → pillai_trace` |
| Re-implement inference / uplift / counterfactual touchpoints on `CausalInference` (and DoWhy where appropriate) | 3 | Includes `infer_do_why.py` consolidation onto DoWhy |
| Local mypy stubs (`typings/pgmpy/*.pyi`) for the surface we use | 1 | Neither pgmpy nor pyAgrum ship `py.typed` |
| Update tests (parity gate first, then existing suite) | 2 | Parity gate first, then the existing suite |
| Pickle / artefact migration | 1 | See strategy below |
| `pyproject.toml` rewrite for uv (already-decided package-manager swap) | 1 | Out of scope for *this* ADR; included for total-cost transparency |
| Docs (MkDocs-Material per causal-analytics convention) | 1 | |
| **Subtotal one-time** | **~13 days** | Risk-adjusted range: **12–22 days** |
| Year-1 maintenance | **3–6 days/yr** | Track pgmpy 1.x minor breaks (1.0→1.1 in 12 mo); own NOTEARS-MLP torch-compat |

**Option (b) — soft-fork baseline (for comparison only)**:

| Workstream | Days |
|---|---|
| Fork to Playtika org, set up CI | 2.5 |
| Drop `torch==2.2.1` / `pyarrow==15.0.0` pins; revalidate against torch ≥2.6 | 3 |
| Patch / re-test against the 6 CVEs | 2 |
| Triage long-tail upstream issues we now own | 1.5 |
| **Subtotal one-time** | **~9 days** |
| **Year-1 maintenance** | **15–30 days/yr** (CVE response, torch breaking-change tracking, semver, downstream user issues) |

Soft-fork is cheaper to *start* and **3–6× more expensive to keep alive**; the curves cross inside year one. Option (d) wins on TCO.

**Pickle / artefact migration strategy**: **hard break with retraining** as part of the 0.2.0 bump. Justification: causalnex's `BayesianNetwork.__reduce__` pickles its full class graph including the abandoned `pgmpy<0.1.20` engine; any forked-or-renamed class breaks the pickle anyway, so a dual-load shim is more code with worse semantics than retraining. Retraining is feasible per problem statement, and 0.2.0 is the natural moment to break artefact format.

## 8. Consequences

**Easier**:

- CVE patch loop reduces to "bump pgmpy/torch", a normal Dependabot flow. The 6-CVE chain is closed at 0.2.0.
- Inference performance: pgmpy's `BeliefPropagation` and `VariableElimination` are actively optimized (256 commits L90D); the 7-min queries are realistic candidates for ≥10× speedup, to be confirmed in Phase B benchmarks on the research-tier worker.
- `do()` and ATE estimation move to a documented, tested API (`CausalInference.query`, `estimate_ate`) rather than the ad-hoc inference shims `causalnex` exposes.
- DoWhy gives us a real causal-effect identification + refutation surface for `infer_do_why.py` (backdoor / IV / frontdoor + `refute_estimate`), which causalnex never had.

**Harder**:

- We own NOTEARS-MLP outright. The upstream reference is dormant; bug fixes are on us.
- `pgmpy` does not ship `py.typed`; we maintain local stubs for the surface we touch.
- pgmpy 1.x is still stabilizing (1.0 → 1.1 in 12 months with breaking renames). We pin tightly and add an integration test that runs against the latest pgmpy before each bump.

**What we lose**:

- causalnex's discretization helpers — re-implement the small set we use (~150 LOC of utilities in our tree, no external dep).
- Pickle compatibility with existing `.pkl` artefacts (see §7).

## 9. Open questions for Phase B

1. Does the parity gate (§6) pass at 1e-6 on both pgmpy and pyAgrum? (Run the spec.)
2. Does pgmpy `VariableElimination`/`BeliefPropagation` close the 7-minute query? Target: ≤30 s, stretch ≤5 s. **Not verified on production data** (decided 2026-10-21 to skip; the data is internal and not to be recorded in this repo). `benchmarks/bench_inference.py` is available for anyone with access to run privately; on the public-example and synthetic networks tested, queries take ≤ 0.23 s.
3. Should `infer_do_why.py` migrate to DoWhy *now* (single 0.2.0 break) or in a follow-on 0.3.0?
4. Local pgmpy stubs: ship in-tree under `typings/pgmpy/` or upstream a `pgmpy-stubs` package contribution?
5. Discretization utilities — keep in-tree or extract as `causal-canvas-discretize` micro-package?
6. Retrain budget: which of the production-deployed `.pkl` artefacts must be retrained before 0.2.0 ships, and who owns each?

## 10. Out of scope

- Real-data benchmarks on production datasets (Phase B).
- Source-code changes, PR plan, branch strategy (Phase B).
- The Poetry → uv swap — already decided pre-ADR; included in §7's day-count for transparency only.
- Replacement of upstream `xunzheng/notears` (we keep the vendored MLP; reference impl is dormant but we are not tracking it).
- Selection of a structure-learning library beyond NOTEARS (e.g. `causal-learn`, `gcastle`) — file as separate ADR if the need arises.

---

**Top 2 risks of this recommendation** (per Objective 0):

1. **pgmpy 1.x is still stabilizing**: 1.0.0 (2025-03) renamed `BayesianNetwork → DiscreteBayesianNetwork`, removed `BayesianModel`/`MarkovModel`/`ContinuousFactor`, renamed `ci_pillai → pillai_trace`. We are migrating into a moving target. Mitigation: tight version-pin + integration test against `latest` before each bump.
2. **NOTEARS ownership is now explicit**: once we drop the causalnex pin, our vendored NOTEARS-MLP must keep up with whatever torch we standardize on, and the upstream `xunzheng/notears` reference is dormant. Mitigation: write down the algorithm provenance and a torch-compat test in Phase B; keep the MLP behind a stable internal interface so a future swap to `causal-learn` or in-house rewrite is contained.

## 11. Implementation notes (2026-10-20)

What was built, and where it departs from the proposal.

**As proposed**

- `causalnex` is no longer a dependency. `torch>=2.13` and `pyarrow>=19` are unpinned from the old chain.
- `causal_canvas/bayesian_network.py` — `BayesianNetwork` over `pgmpy.models.DiscreteBayesianNetwork` (MLE / K2 / BDeu through `pgmpy.parameter_estimator`); keeps the `cpds` / `predict` / `predict_probability` surface.
- `causal_canvas/inference_engine.py` — `InferenceEngine` over `VariableElimination`; `query`, soft `do_intervention`, `reset_do`. `query(nodes=[...])` restricts the work to the requested nodes.
- `causal_canvas/vendored/` — the causalnex pieces we use, owned in-tree with upstream headers and licence (`StructureModel`, linear NOTEARS, `LocallyConnected`, `DistTypeBase` and distribution types, discretisers, plots).
- §6 parity gate: `tests/test_bayesian_network_pgmpy.py` checks marginal, conditional and interventional queries against brute-force enumeration (1e-6). The pgmpy half of the gate is implemented; the pyAgrum cross-check was not (pyAgrum was not adopted).
- Hard break for pickles: `safe_pickle_load` raises a "retrain" error for pre-0.2.0 artefacts.
- DoWhy is an optional extra (`causal-canvas[dowhy]`) with `causal_canvas.effect_estimation.estimate_effect`.

**Deviations and findings**

1. **pgmpy 1.1 requires `numpy>=2`** (and DoWhy 0.14 `numpy>2`), which the ADR did not anticipate. The dependency floor therefore moved with it: `pandas>=2.2.2`, `plotnine>=0.14`, `numpy>=2`, Python `>=3.10,<3.13`. This is part of the 0.2.0 break.
2. **§9 Q3 (when to move `infer_do_why.py` to DoWhy):** decided *not now*. `infer_do_why.py` keeps its distribution-level counterfactual / shift workflow, now on pgmpy. DoWhy is available as a separate function; wiring it into the script is a 0.3.0 follow-up.
3. **MDLP discretiser:** `mdlp-discretization` was an optional causalnex extra (and absent from the lockfile). It is now the optional extra `causal-canvas[mdlp]`.
4. **Inference semantics (unchanged from causalnex, now documented):** `predict` / `predict_probability` return `P(node | parents)` straight from the CPT when all parents are present (other columns are ignored), and otherwise exact `P(node | all observed columns)` by variable elimination -- not pgmpy's joint MAP over all unobserved nodes. The two regimes are deliberately different; use `InferenceEngine` for full-evidence posteriors. Interventions are applied by graph surgery on a copy of the pgmpy model.
5. **pgmpy API drift inside 1.x:** `pgmpy.estimators.{MaximumLikelihoodEstimator,BayesianEstimator}` already warn as deprecated (removal in 1.3); we use `pgmpy.parameter_estimator` and pin `pgmpy>=1.1.0,<1.2.0`.

**Fact refresh (2026-10-20)** — the ADR was written 2026-04-25; re-checked against PyPI/GitHub before implementing:

- `causalnex` 0.12.2 (2026-08-31) is labelled "final release" upstream; issue #242 is still unanswered. It dropped the `torch==2.2.1` pin (`torch>=1.7`) but still requires `numpy<1.24`, `pandas<2.0`, `pgmpy<0.1.20`, Python `<3.11`. The decision stands; the blocker is now the numpy/pandas/pgmpy ceiling rather than torch/pyarrow.
- pgmpy is at 1.1.2 (2026-04-30), still inside the `<1.2` pin; DoWhy 0.14 is still the latest; pyAgrum is now 3.x (not adopted).
- torch 2.12.1 is affected by CVE-2025-3000 (fixed in 2.13.0), so the floor is `torch>=2.13`.

**Numeric comparison against causalnex 0.12.1 (2026-10-21)** — same discretised data and DAG (students example), old stack (pandas 1.5.3 / pgmpy 0.1.19 / causalnex 0.12.1 on `develop`) vs this branch:

- Fitted CPDs for G1, G2, G3, `higher`: byte-identical CSVs. Discretiser split files: identical.
- Marginals for all 46 nodes: max |diff| 1e-15. Interventions (shifts, ATEs, uplifts) and counterfactuals: max |diff| ≤ 3e-16.
- `evaluate-experiments`: `thresholds.json`, test/optim predictions and bootstrapped scores identical.
- `infer-dowhy` end to end: ~75 s on causalnex vs ~4 s here (including interpreter and import time); this is the only timing from a real model, still on a small network.
- One regression found and fixed by this comparison: numpy 2 scalar reprs (`np.int64(0)`) leaked into CSV index labels; node states are now native Python scalars.
- Not covered: structure learning (NOTEARS-MLP is stochastic across torch versions, so only the BN stage was compared on a fixed DAG) and production-sized models.
