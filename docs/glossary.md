# Glossary

Project-wide glossary. The PM-friendly version is at [for-pms/glossary.md](for-pms/glossary.md); this page adds entries that matter mostly to analysts and statisticians.

| Term | Definition |
|---|---|
| **A/B test** | Randomised controlled experiment. Gold standard for causal claims. Causal Canvas estimates causation from observational data, which is strictly weaker. |
| **ATE** | Average Treatment Effect — $\\mathbb{E}[Y \\mid \\mathrm{do}(X = x_1)] - \\mathbb{E}[Y \\mid \\mathrm{do}(X = x_0)]$. Population-level average effect of an intervention. |
| **Back-door criterion** | A graphical condition (Pearl) under which an intervention's effect is identifiable from observational data given a specific adjustment set. |
| **Bayesian estimator** | An estimator that adds a prior (K2 or BDeu) to the maximum-likelihood estimator of CPT cells. Used by Causal Canvas to regularise sparse cells. |
| **Bayesian network (BN)** | A DAG plus a probability distribution at each node; together encode a joint distribution that factorises along the graph. |
| **Bootstrap** | Resampling-based estimation of confidence intervals. Used by `evaluate-experiments` to score F1 / precision / recall across models. |
| **Causal discovery** | Recovering a DAG from observational data. NOTEARS is the discovery method used here. |
| **Causal sufficiency** | Assumption that all common causes of any pair of measured variables are themselves measured (no unmeasured confounders). |
| **CATE** | Conditional Average Treatment Effect — ATE within a sub-population. Not directly emitted by Causal Canvas; obtain by conditioning the BN. |
| **Confounder** | A variable that affects both supposed cause and supposed effect; ignoring it biases the estimated effect. |
| **CPD / CPT** | Conditional Probability Distribution / Table — the probability table at each node in a Bayesian network. |
| **DAG** | Directed Acyclic Graph. Output of causal discovery. Encodes "A causes B" with no cycles. |
| **Discretisation** | Mapping a continuous variable to a finite set of bins. Necessary for the discrete BN fit; introduces bias (see [Discretisation bias](for-statisticians/discretisation-bias.md)). |
| **Do-operator** | $\\mathrm{do}(X = x)$. Causal-inference operator that severs incoming edges to $X$ and forces it to $x$. |
| **Faithfulness** | Assumption that conditional independencies in the data exactly match the d-separations of the DAG. |
| **Identifiability** | Whether a causal quantity (e.g. ATE) can be expressed in terms of observational quantities under the model's assumptions. |
| **Intervention** | An action that fixes a variable's value, modelled by the do-operator. Different from observational conditioning. |
| **Markov compatibility** | Property of a distribution factorising along a DAG: $P(\\mathbf{X}) = \\prod_i P(X_i \\mid \\mathrm{Pa}(X_i))$. |
| **MLE** | Maximum Likelihood Estimator. The "count and divide" CPT estimator. |
| **NOTEARS** | "DAGs with NO TEARS" — the 2018 algorithm Causal Canvas uses for structure learning, framed as a continuous-optimisation problem. |
| **Pruning / w_threshold** | Dropping edges with weight below a threshold $w$ to recover a discrete sparse DAG from the dense NOTEARS solution. |
| **Structural causal model (SCM)** | A set of structural equations $X_i = f_i(\\mathrm{Pa}(X_i), \\epsilon_i)$ defining the data-generating process. The DAG is its qualitative skeleton. |
| **Tabu edges** | Edges the structure-learning algorithm is forbidden from drawing, encoding prior direction knowledge. |
| **Tweedie** | A flexible exponential-family distribution; useful for revenue-like variables with a mass at zero plus a continuous tail. Implemented at scale 1.5 in `custom_distributions/tweedie.py`. |
