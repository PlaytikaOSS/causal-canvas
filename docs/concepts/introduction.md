<!-- This page was ported from the GitHub wiki. -->

# Introduction

Causal inference is crucial for businesses as it enables a deeper understanding of the underlying relationships between variables that cannot be obtained through predictive models alone. While predictive models are valuable for forecasting and predicting outcomes, they do not provide insights into the causes behind those outcomes. Causal inference goes beyond correlation and uncovers the mechanisms and factors that drive specific behaviors or outcomes in a business setting.

One significant application of causal inference in Playtika could be the ability to identify the triggers of churn or sink behaviour etc. By studying causal relationships, we can determine the factors that lead to customer attrition or disengagement, allowing us to take proactive measures to prevent it. This understanding can provide actionable insights to design effective retention strategies or targeted interventions.

Another advantage of incorporating causal inference is its ability to measure the true impact of interventions or treatments through rigorous A/B testing. While A/B tests can quantify the average treatment effect, causal inference offers a more comprehensive approach to uncovering the causal links between interventions and outcomes. This knowledge can help us to take informed decisions about which treatments or strategies are most effective and how to design new treatments tailored to specific goals.

# Summary of Steps in the Causal Inference Methodology:

![Diagram: causal process](https://github.com/PlaytikaOSS/causal-canvas/raw/main/images/causal_process.png)
**Figure 1:** Process of causal canvas methodology

We utilised the causalnex library (1), a Python library for causal inference, to implement the methodology and perform the necessary computations. We perform the following steps on a separate validation set to ensure unbiased evaluation:

1. Assume Causal Restrictions: Make certain assumptions about causal relationships between variables to guide the analysis. These are:
    a. Ensure Target-Feature Causality: Confirm that the target variable does not cause any of the features. For example, in the context of churn prediction, ensure that churn cannot cause time spent on the game.
    b. Ensure Feature Restriction: Verify that no feature can cause the value of the "days_since_install" variable at the last feature date.

2. Fit the SEMs:
   a. Assume (for now) a linear system of equations.
   b. Apply L1 Regularization: Utilize L1 regularization on the fully connected graph to control for spurious connections and enhance model interpretability.

3. Optimize Weights: Optimize the weights of the model to find the best fit based on the given assumptions and restrictions.

4. Hard Thresholding: Apply a hard threshold to the weights to eliminate false discoveries and focus on significant causal relationships.

5. Utilize NOTEARS Algorithm: Employ the NOTEARS algorithm, a specific methodology for learning the structure of causal networks, to guide the analysis.

6. Threshold Graph to DAG: Continue iteratively thresholding the graph until it becomes a Directed Acyclic Graph (DAG) or a Bayesian Network, which represents the causal relationships among variables.

7. Discretize Numerical Values: Convert numerical values into discrete categories to better handle them within the causal inference framework.

8. Choose Bayesian Network Estimation Method: Apply either Maximum Likelihood (ML) or Bayesian estimation techniques to estimate the parameters of the causal model based on the available data.

9. DoWhy experimentation



# Reasons why causalnex is a fully comprehensive library for causal inference

![Diagram: pgmpy%20paper](https://github.com/PlaytikaOSS/causal-canvas/raw/main/pgmpy%20paper.png)
**Figure 2:** Summary of different causal libraries existing in python. Taken from pgmpy paper (2).

References

(1) https://causalnex.readthedocs.io/en/latest/01_introduction/01_introduction.html

(2) https://arxiv.org/pdf/2304.08639.pdf → pgmpy paper

(3) https://github.com/pgmpy/pgmpy → pgmpy githugb

(4) https://pgmpy.readthedocs.io/en/latest/ → pgmpy documentation
