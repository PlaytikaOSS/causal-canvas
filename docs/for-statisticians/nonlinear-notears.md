# Non-linear NOTEARS

The original NOTEARS algorithm (Zheng et al., 2018) parameterises the structural equations as linear: $X_j = \\sum_i W_{ij} X_i + \\epsilon_j$. Causal Canvas extends this to non-linear equations via the MLP variant of Zheng et al. (2020), with Playtika-specific distribution types layered on top through the `causalnex` `DistTypeBase` interface.

## Continuous-optimisation formulation (recap)

NOTEARS reformulates DAG learning as a continuous optimisation problem over a real-valued weighted-adjacency matrix $W$:

$$
\\min_W \\; \\ell(W; X) \\quad \\text{s.t.} \\quad h(W) = \\mathrm{tr}(e^{W \\circ W}) - d = 0
$$

The acyclicity constraint $h(W) = 0$ is enforced by augmented Lagrangian. The weight magnitudes after convergence are pruned at `w_threshold` to recover a discrete DAG.

For the linear case, the loss $\\ell$ is the residual sum of squares. For the non-linear MLP case (Zheng et al., 2020), each variable's structural equation is parameterised as $X_j = f_j(\\mathrm{Pa}(X_j); \\theta_j)$ where $f_j$ is a small MLP. The first layer's input weights, $\\|W_j^{(1)}\\|_2$ across input units, serve as the edge-weight proxy that goes into $h(W)$.

## Distribution-type extensions

`causal_canvas.vendored.dist_type` (vendored from causalnex) provides a registry of parametric output distributions per node, allowing the loss to be the appropriate negative log-likelihood per variable type instead of MSE everywhere. Causal Canvas adds:

- **`bin`** — Bernoulli / binary cross-entropy (vendored from causalnex).
- **`cat`** — Categorical / cross-entropy (vendored from causalnex, refined by `causal_canvas.custom_distributions.categorical`).
- **`cont`** — Gaussian / MSE (vendored from causalnex).
- **`pos_cont`** — non-negative Gaussian (`custom_distributions.non_negative_continuous`); enforces a softplus-transformed mean.
- **`bimodal_pos_cont`** — mixture of two non-negative Gaussians (`custom_distributions.non_negative_bimodal`); useful for skewed engagement metrics.
- **`ord`** — ordinal regression (vendored from causalnex); preserves rank structure.
- **`poiss`** — Poisson (vendored from causalnex); for count features.
- **`gamma`** — Gamma (`custom_distributions.gamma`); for strictly-positive heavy-tailed continuous features.
- **`tweedie`** — Tweedie (`custom_distributions.tweedie`) at scale factor 1.5; mass at zero plus continuous tail, common for revenue.

Each entry is a class registered through the `DistTypeBase` distribution-type interface, providing `loss`, `inverse_link_function`, and (where applicable) reparameterisation helpers.

## Configuration

Set `connections_type: non_linear` and provide `non_linear_args.feature_to_distribution_map` mapping each column to one of the codes above. See [Configuration reference](../reference/config.md).

## Caveats

- Non-linear NOTEARS-MLP is **not deterministic**: the MLP weights are randomly initialised. Re-run with several seeds and keep edges that appear in ≥ 80% of runs.
- The MSE-equal-variance failure mode that affects linear NOTEARS (Reisach et al., 2021) is reduced but not eliminated by the MLP. Standardise inputs and use distribution-appropriate loss types where you can.
- Larger MLPs (`hidden_layer_units = [16, 16]` and beyond) over-fit on small samples and produce dense graphs. Start small (`[2]` or `[4]`) and grow only if the discovered graph is implausibly sparse.

## See also

- [Assumptions](assumptions.md).
- [References](references.md).
- Module-level docstrings: `causal_canvas/nonlinear_notears.py` (high-level API) and `causal_canvas/custom_notears_mlp.py` (low-level solver).
