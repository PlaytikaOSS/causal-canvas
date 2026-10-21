<!-- This page was ported from the GitHub wiki. -->

To understand causality, researchers and analysts employ a variety of approaches, including the computation of Average Treatment Effect (ATE), Conditional Average Treatment Effect (CATE), Counterfactuals, and Uplift. These concepts provide essential tools for quantifying and evaluating the effects of interventions, helping decision-makers make informed choices and optimise outcomes.

Each of these concepts serves a distinct purpose in causal inference:

## ATE

The Average Treatment Effect (ATE) is a measure used in causal inference to quantify the average difference in outcomes between a treatment group and a control group. Mathematically, it can be represented as:

<!-- legacy GitHub asset omitted: Inline figure (legacy GitHub asset) -->
where:

- Y_1​ is the potential outcome when the treatment is applied.
- Y_0​ is the potential outcome when no treatment is applied.
- E represents the expected value.

![Diagram: Intervention](https://github.com/PlaytikaOSS/causal-canvas/raw/main/images/Intervention.png)

It helps answer questions like, "On average, does the treatment work?" This is essential for making high-level decisions and resource allocation.

## CATE

Conditional Average Treatment Effect (CATE): CATE goes beyond ATE by allowing us to assess the treatment effect within specific subgroups of the population. This helps tailor interventions to individuals with certain characteristics, ensuring a more personalized and effective approach. Mathematically, it can be represented as:
![Diagram: CATE](https://github.com/PlaytikaOSS/causal-canvas/raw/main/images/CATE.png)

Where X represents a set of covariates or characteristics.

## Counterfactuals

Counterfactuals enable us to explore what might have happened if a different treatment or action had been taken. By comparing actual outcomes with these hypothetical scenarios, we gain insights into causality.

Counterfactuals are hypothetical scenarios used in causal inference. They represent what would have happened if a different treatment or intervention had been applied. In the context of ATE and CATE, counterfactuals are the unobserved potential outcomes (Y_1​ and Y_0​) that allow us to compare treatment effects. By comparing actual outcomes with these hypothetical scenarios, we gain insights into causality but are not directly observable. In the context of ATE and CATE:

Y​ represents outcome under treatment.
Y'​ represents outcome without treatment.

![Diagram: COUNTERFACTUALS](https://github.com/PlaytikaOSS/causal-canvas/raw/main/images/COUNTERFACTUALS.png)

## Uplift

Uplift, also known as "lift" or "treatment effect," quantifies the change in outcomes caused by a specific treatment or intervention. It is often used in marketing and personalized medicine to determine the incremental impact of an action. Mathematically, uplift can be represented as:

![Diagram: UPLIFT](https://github.com/PlaytikaOSS/causal-canvas/raw/main/images/UPLIFT.png)

X is a binary treatment indicator or a level indicator.
How to calculate these quantities with Bayesian Networks

After conducting inference with Bayesian Networks and estimating parameters using methods like Maximum Likelihood Estimation (MLE), K2 prior, or BDeu prior, we can calculate ATE and CATE as follows:

![Diagram: eq 4 1](https://github.com/PlaytikaOSS/causal-canvas/raw/main/images/eq_4_1.png)

### MLE

We can straightforwardly use the maximum likelihood estimates and calculate the ATE and CATE(x) where x is the level of a variable.

### K2 or BDEU prior

These are typically used for parameter estimation in Bayesian Networks rather than directly estimating ATE or CATE. However, you can use the estimated parameters to calculate ATE and CATE in the same way as MLE.

## Extensions of treatment effects

### Odds Ratio (OR)

The Odds Ratio can be an extension of ATE and CATE and is often used in epidemiology and medical research. It measures the likelihood of an event occurring in the presence of a treatment or exposure compared to the likelihood in the absence of the treatment or exposure. It can provide valuable insights into the association between a treatment and an outcome, especially in studies involving binary outcomes.

The formula for the OR is:
![Diagram: eq 4 2](https://github.com/PlaytikaOSS/causal-canvas/raw/main/images/eq_4_2.png)

### Relative Risk (RR)

Relative Risk can be an extension of ATE and CATE and is commonly used in epidemiology as well. It measures the ratio of the probability of an event occurring in the treated group to the probability in the control group. It helps us quantify how much the treatment changes the risk of an event happening and is particularly useful for assessing the impact of treatments on disease outcomes.

The formula for the RR is:

![Diagram: eq 4 3](https://github.com/PlaytikaOSS/causal-canvas/raw/main/images/eq_4_3.png)

### Risk Difference (RD)

The Risk Difference (RD), also known as the Absolute Risk Reduction, is conceptually the same as both the Average Treatment Effect (ATE) and the Conditional Average Treatment Effect (CATE) in many situations. It quantifies the absolute difference in outcomes between treated and untreated groups, taking into account the baseline or control group outcome. The formula for the Risk Difference is:

![Diagram: eq 4 4](https://github.com/PlaytikaOSS/causal-canvas/raw/main/images/eq_4_4.png)

It represents the absolute change in outcomes due to the treatment. It's particularly useful when you want to express the treatment effect in terms of the actual units of measurement, making it easier to interpret and communicate to a broader audience. In this sense, it's similar to ATE, which provides a global measure of the average treatment effect, and CATE, which assesses treatment effects within specific subgroups.

## More resources
https://fabiandablander.com/r/Causal-Inference.html
https://www.ucl.ac.uk/lagnado-lab/publications/lagnado/intervention%20hagmayer%20et%20al.pdf
https://web.stanford.edu/~doubleh/lecturenotes/slides2_summer2012.pdf
https://www.uplift-modeling.com/en/latest/user_guide/introduction/cate.html
https://imai.fas.harvard.edu/teaching/files/ate.pdf
https://www.youtube.com/watch?v=pg7P-BsHQ_k
https://www.statology.org/confidence-interval-for-odds-ratio/
https://sphweb.bumc.bu.edu/otlt/mph-modules/bs/bs704_confidence_intervals/bs704_confidence_intervals7.html
https://sphweb.bumc.bu.edu/otlt/mph-modules/bs/bs704_confidence_intervals/bs704_confidence_intervals8.html