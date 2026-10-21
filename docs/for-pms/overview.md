# What Causal Canvas does for product decisions

## In one sentence

Causal Canvas reads your existing player-360 data and proposes which levers — features, campaigns, mechanics — most likely move a metric you care about, like churn or spend.

## What you get back

1. A **graph** of which variables influence which: arrows mean "A probably affects B".
2. A **ranking** of levers by estimated impact on the target metric.
3. A **what-if calculator**: "If we push 70% of users into the low-ads segment, churn probability moves from 18% to 14%."

## What you can decide with it

- Which feature changes are worth A/B-testing first.
- Which user segments to focus retention or monetisation work on.
- Which "common-sense" assumptions about your game are *not* supported by the data (often the most useful output).

## What you cannot get back

- **Proof of causation.** The tool estimates causality under assumptions; an A/B test is still the gold standard.
- **Effects of things you did not measure.** If a major driver is absent from the data, it is absent from the answer too.
- **Future effects of a feature that didn't exist in the data.** Causal Canvas only knows what was in your training window.

## Risk and limits

- The output looks decisive (numbers, arrows, plots). Treat it as a *prioritised list of hypotheses to test*, not as evidence.
- The model assumes the causal structure is stable over the period you trained it on. If the game changed midway through (a major release, a price change), retrain on the new period.
- An estimated intervention effect is only as good as the assumption that no important confounders are missing. Read [Questions to ask your data team](questions-to-ask.md) before acting on results.

## Going deeper

- :material-clipboard-list: [Questions to ask your data team](questions-to-ask.md)
- :material-book-open-variant: [Glossary](glossary.md)
- :material-chart-line: [How analysts use this tool](../for-analysts/index.md)
