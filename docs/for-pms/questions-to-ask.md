# Questions to ask your data team

A Causal Canvas output looks crisp: a graph, a ranked list of drivers, a what-if calculator. The questions below help you separate signal from artefact before acting.

## Before trusting the graph

- [ ] **Which variables were excluded?** And why? Variables that were available but dropped are usually the most informative ones to ask about.
- [ ] **Is there a known confounder we can't measure?** (E.g. "user mood", "external marketing campaigns we don't track"). If yes, the estimated effects on adjacent variables are biased — by how much, nobody knows.
- [ ] **How stable is the graph across time windows?** If the same data window split in half produces two different graphs, the result is not robust enough to act on.
- [ ] **Does the graph break common sense?** If churn appears to "cause" first-day login, the data direction or the labelling is wrong.

## Before trusting an intervention estimate

- [ ] **What does the tool say about levers we already A/B-tested?** Direction agreement is reassuring; magnitude disagreement is normal but worth understanding.
- [ ] **What's the bin granularity for the lever you want to change?** A 3-bin "low / medium / high" intervention answers a different question than a continuous shift.
- [ ] **What's the assumed counterfactual baseline?** "Increase X" relative to what — current population mean, last quarter, a specific segment?

## Before bringing it to the steering committee

- [ ] **Is the recommendation falsifiable in an A/B test?** If yes, that test is the next step, not the rollout.
- [ ] **What's the confidence around the rank order?** If "feature A" and "feature B" estimates are within noise of each other, treat the ranking as a tie.
- [ ] **What part of the analysis should we redo if the data refreshes?** Most graphs need to be re-fit when the underlying data window moves by more than a few weeks.

## Red flags

- The model says a feature you launched two weeks ago has the largest effect. (It probably picked up a temporary novelty effect.)
- The graph is fully connected with no clear root causes. (Likely under-regularised; ask your team to bump `w_threshold`.)
- The same input config produces different graphs on consecutive runs. (Non-determinism — the structure isn't stable enough to act on.)
