# Glossary (PM-friendly)

Plain-language definitions of the terms you'll hear in a Causal Canvas review.

| Term | Plain-language definition |
|---|---|
| **Causal discovery** | Learning from data which variable influences which. The output is a graph of arrows. |
| **DAG** | "Directed Acyclic Graph". A graph of arrows with no loops. Used to encode "A causes B causes C" without contradictions. |
| **Bayesian network** | A DAG plus a probability table at every node. Lets you ask "given these inputs, what is the chance of churn?". |
| **CPT / CPD** | "Conditional Probability Table" / "Distribution". The probability table attached to each node in a Bayesian network. |
| **Discretisation** | Splitting a numeric column (e.g. "balance") into bins so the model can group similar users. |
| **Tabu edges** | Arrows the algorithm is forbidden from drawing because they don't make causal sense (e.g. "churn caused user's age"). |
| **Intervention** | "What if we forced X to value x?" — different from observing X = x because it ignores all the reasons X usually takes that value. |
| **Counterfactual** | "Given what we saw for this user, what *would have* happened if X had been different?" |
| **ATE** | "Average Treatment Effect" — the average change in the outcome from an intervention across the whole population. |
| **NOTEARS** | The 2018 algorithm Causal Canvas uses to learn the DAG. Just a name; you don't need to know how it works to read the output. |
| **Confounder** | A third variable that affects both the supposed cause and the supposed effect. Ignoring it makes the estimated effect biased. |
| **A/B test** | The gold-standard way to measure causation. Causal Canvas estimates causation from observational data, which is weaker. |
| **Observational data** | Data from users behaving naturally, without any experiment forcing some users into one group and others into another. |

## Going deeper

- The full glossary (with statistician-grade definitions) is at the [project glossary](../glossary.md).
- The [analyst pitfalls](../for-analysts/pitfalls.md) page covers the common ways these terms get mis-used.
