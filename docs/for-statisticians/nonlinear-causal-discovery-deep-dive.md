<!-- This page was ported from the GitHub wiki. -->

# Non‐linear Causal Discovery
## Brief refresher of Linear Causal Discovery with NOTEARS

Remember the process of Linear Causal Discovery with NOTEARS, as described in [2. Linear Causal Discovery](./linear-causal-discovery.md). 

As seen in Figure 1, a system of Structural Equation Models was used, assuming linear relationships between features and additive noise centered at 0.

![Diagram: non linear](https://github.com/PlaytikaOSS/causal-canvas/raw/main/images/non_linear.png?raw=true
)
**Figure 1:** Summary of Linear Causal Discovery with NOTEARS

## Why linearity assumption can hurt

Assuming linear structural equation models (SEM) for causal inference can fail to capture non-linear relationships, potentially leading to overlooked or misrepresented causal connections in the underlying data generating process. Linear SEMs are limited by their inherent linearity assumption, and when the true relationships between variables are non-linear, the model may inadequately represent the complexity of the causal structure. This limitation can result in the omission of significant non-linear patterns, leading to the failure to detect crucial relationships or, conversely, the creation of false edges in the causal graph. In causal inference, accurately capturing the underlying mechanisms necessitates considering non-linearities, and relying solely on linear SEMs can thus compromise the validity and completeness of causal models, potentially leading to biased or misleading results.

To give an illustration, in Figure 2, in the first example where the ground truth is a causal relationship A->B with a non-linear nature, but the linear model estimates suggest that A and B are not causally related (correlation between A and B is 0), this discrepancy can arise due to the limited expressive power of linear models. Linear models assume a constant linear effect, and when the true relationship between A and B is non-linear, the linear model fails to capture this complexity. As a result, the estimated correlation might be close to zero, leading to a false inference of no causal relationship.

In the second example where the ground truth involves a causal chain A->B->C with all non-linear relationships, but the linear model estimates suggest both A->B->C and A->C (correlation between A and C conditioned on B is non-zero), this discrepancy highlights the challenge of disentangling direct and indirect effects in linear models. Linear models may mistakenly attribute some of the indirect effects to direct relationships, especially when non-linear dependencies exist. The correlation between A and C conditioned on B being non-zero indicates a potential false inference of a direct causal relationship A->C, emphasising the limitations of linear models in capturing complex, non-linear causal structures.

![Diagram: causal linear non linear nodes](https://github.com/PlaytikaOSS/causal-canvas/raw/main/images/causal_linear_non_linear_nodes.png)
**Figure 2:** An example of causally related features cannot be detected by linearity assumptions



## How to solve this

The same authors of (1) introduce a novel framework in (2) for addressing the challenges associated with learning sparse nonparametric directed acyclic graphs (DAGs) from data. Their approach is rooted in the same algebric characterization of DAGs, which led to the development of a continuous program for score-based learning of DAG models parametrised by linear structural equation models (SEM). To extend this characterisation to nonparametric SEM, the authors leverage nonparametric sparsity through partial derivatives. This results in a continuous optimisation problem that is versatile enough to be applied to various nonparametric and semi-parametric models, including but not limited to generalised linear models (GLMs), additive noise models, and index models. Unlike existing methods that often require specific modeling choices, loss functions, or algorithms, the proposed framework is entirely general, accommodating general nonlinear models, diverse differentiable loss functions, and generic black-box optimisation routines.

We will focus on the Generalised Linear Models and explain how the authors solve this issue, and what approaches we used within Playtika.

# Non-parametric NOTEARS-MLP methodology

For a feature set X consisting of p variables, including the target variable, we aim to estimate the existence and nature of causal relationships X_i​→X_j​. By considering potentially different or similar distributions for each X, our objective is to minimize the loss function L(f) subject to the constraint that the directed acyclic graph (DAG) constraints G(f) are satisfied, i.e.,

![Diagram: eq 2 1](https://github.com/PlaytikaOSS/causal-canvas/raw/main/images/eq_2_1.png) (1)

The loss function is defined as the average of log-likelihood terms for each variable. To address this optimisation problem, they propose to approximate the functions f_j bu using Multilayer Perceptrons (MLPs) with h hidden layers and a single activation function σ:R→R. The MLP formulation is given by 


![Diagram: eq 2 2](https://github.com/PlaytikaOSS/causal-canvas/raw/main/images/eq_2_2.png)(2)

The proposed model ensures acyclicity and identifiability in the learned causal relationships.

Given a j-th Multilayer Perceptron (MLP) represented by parameters ![](
<!-- legacy GitHub asset omitted: image -->
), all parameters are collectively defined as ![Diagram: eq 2 4](https://github.com/PlaytikaOSS/causal-canvas/raw/main/images/eq_2_4.png) and the weight matrix W can be expressed as ![Diagram: eq 2 5](https://github.com/PlaytikaOSS/causal-canvas/raw/main/images/eq_2_5.png), i.e., 
![Diagram: proposition non linear](https://github.com/PlaytikaOSS/causal-canvas/raw/main/images/proposition_non_linear.png)

The optimisation program then aims to minimise 

![Diagram: eq 2 6](https://github.com/PlaytikaOSS/causal-canvas/raw/main/images/eq_2_6.png)(3)

Similar to the NOTEARS algorithm, the augmented Lagrangian method is employed with L-BFGS for optimisation. This formulation ensures that the weight matrices are sparse while capturing the nonparametric relationships in the data.

## Generalised Linear Models Representation

![Diagram: glm table](https://github.com/PlaytikaOSS/causal-canvas/raw/main/images/glm_table.png)
**Figure 3:** Equations of ANN and GLMs and link functions for GLMs 


As we can see from Figure 3, in a NANM, we assume that the relationship between the input features and the target variable involves an underlying non-parametric function (meaning, not predefined by a specific equation), and any deviations from this function are treated as random noise. The model captures the main trend in the data without making strong assumptions about its exact form. On the other hand, GLMs are a type of models designed for specific types of data (like binary, count, or continuous). It assumes a linear relationships between the input features and a linear combination of parameters, with a link function transforming this combination to the expected value of the response variable. The "generalized" part means it's versatile enough for various data types.

## Connection to MLP
Both NANMs and MLPs aim to capture the complexity in the data that simple models might overlook. While a NANM focuses on the non-parametric aspect, an MLP is a flexible neural network that can adapt and learn complex relationships. The NANM provides a starting point, and the MLP refines the model by learning intricate patterns.
GLMs capture linear relationships, but they might miss complex, non-linear patterns. NANMs, and subsequently MLPs, can help capture these non-linearities. You can think of a GLM as the initial approximation, and the nonparametric model (NANM/MLP) adds the complexity that might be missing.

## Combining with MLP

For combining NANMs and MLPs, think of the nonparametric part as capturing the complex patterns in the data that a simple equation might miss. MLPs are great for this task because they can learn intricate relationships without being restricted to a predefined structure. For GLMs, since they are powerful, but still have limitations, especially when dealing with intricate relationships or non-linearities, by using a nonparametric model like an MLP, we enhance the GLM's ability to capture these complexities. This combination allows us to benefit from the interpretability of GLMs while harnessing the flexibility of nonparametric models.

Thus, this hybrid statistical process involves starting with GLM link functions to capture basic linear relationships and then incorporating them within a complex nonparametric model like MLPs to enhance the model's capability we can capturing complex patterns and non-linearities present in the data while we still extract some interpretation. This combination offers a robust and flexible approach, especially for GLMs applied in various scenarios.



![Diagram: non linear process](https://github.com/PlaytikaOSS/causal-canvas/raw/main/images/non_linear_process.png)
**Figure 4:** Flow of how a mixture of GLMs and MLP works for non-linear causal discovery with NOTEARS.


# Functionality within causal_canvas and causalnex

Causalnex provides the MLP process via dual ascending steps, as taken from the authors of (2). They had developed a few customised GLM functions but we incorporated a few changes within the main MLP class as well.

| **Distribution**              | **Tool alias**    |**Causalnex Library Included**| **Additions Developed** | **Tested**  |
|-------------------------------|-------------------|----------------------------|---------------------|---------|
| Binary Cross Entropy          | bin               | Yes                        | -                   | -       |
| Poisson                       | poiss             | Yes                        | -                   | -       |
| Gaussian                      | cont              | Yes                        | -                   | -       |
| Multinomial/Categorical       | cat               | Yes                        | -                   | -       |
| Ordinal                       | ord               | Yes                        | -                   | -       |
| Gamma                         | gamma             | No                         | Yes (Developed)     | -       |
| Tweedie (scale = 1.5)         | tweedie           | No                         | Yes (Developed)     | -       |
| Gaussian (Constrained to >=0) | pos_cont          | No                         | Yes (Developed)     | -       |
| Bimodal Gaussian Mixture      | bimodal_pos_cont  | No                         | Yes (Developed)     | Pending |

**Table 1:** GLM link functions within causalnex and our contribution

**Additional Functionalities**:

- Loss history tracking.
- Fetching standard errors of the estimates.
- Hessian matrix computation.

These enhancements broaden the scope of causalnex's GLM capabilities by including distributions like Gamma, Tweedie, Gaussian constrained to >=0, and Bimodal Gaussian mixture. Moreover, the incorporation of loss history tracking and standard error retrieval provides users with valuable insights into the learning process and uncertainty estimates, contributing to a more comprehensive and informative modelling experience.

# How it could work for a use-case
![Diagram: config non linear example](https://github.com/PlaytikaOSS/causal-canvas/raw/main/images/config_non_linear_example.png)
**Figure 5:** Snapshot of the configuration file for students dataset within our causal inference tool

Select the appropriate distribution of the scaled features (via manual data exploration)
Select the target distribution, i.e. higher is binary, absences is tweedie etc.
Select lasso_multiplier, rigde_multiplier, whether to model the bias, number of layers with number of neurons and wether to use GPUs.

![Diagram: process example non linear students](https://github.com/PlaytikaOSS/causal-canvas/raw/main/images/process_example_non_linear_students.png)

**Figure 6:** Flow of how a mixture of GLMs and MLP works for non-linear causal discovery with NOTEARS for churn use-case. * biases are omitted for the sake of example. Of course, for a binary feature, such as Pstatus, then the equation is be written wrt log(Pstatus=1/Pstatus=0) etc.

![Diagram: grades linear](https://github.com/PlaytikaOSS/causal-canvas/raw/main/images/grades_linear.png)
**Figure 7:**
Toy causal graph after linear causal discovery with lasso_multiplier =  0.2 and w_threshold=0.08.

![Diagram: grades non linear](https://github.com/PlaytikaOSS/causal-canvas/raw/main/images/grades_non_linear.png)

**Figure 8:** Toy causal graph after non-linear causal discovery with lasso_multiplier =  0.2, ridge_multiplier: 0.1, and one hidden layer with 2 neurons and w_threshold=0.08.

![Diagram: all weights w threshold heatmap](https://github.com/PlaytikaOSS/causal-canvas/raw/main/dummy_example_results/students_portuguese_grades/non_linear_causal_discovery/20240216_230308/causal_discovery/all_weights_w_threshold_heatmap.png)

**Figure 9:** Toy heatmap with estimated weights after thresholding for lasso_multiplier =  0.2, ridge_multiplier: 0.1, and one hidden layer with 2 neurons and w_threshold=0.08. 

| **Directions of features towards G3** | **W** | **s.e.(W)** | **95% low** | **95% high** | **Impact G3** |
|--------------------------------------|-------|-------------|-------------|--------------|---------------|
| **G1**                               | 3.154231 | 0.6055045 | 2.5487265 | 3.7597355 | improve |
| **G2**                               | 14.281498 | 3.7818108 | 10.4996872 | 18.0633088 | improve |

**Table 2:** Toy example of estimated weights, their standard errors and 95% CI after thresholding for lasso_multiplier =  0.2, ridge_multiplier: 0.1, and one hidden layer with 2 neurons and w_threshold=0.08.


# Proposed extensions

Exploring extensions within the Generalized Linear Model (GLM) framework and general functional scope opens avenues for enhanced modeling flexibility and improved representation of complex relationships. Three notable extensions to consider are:

## Polynomials:

Scope within GLM: Introducing polynomial terms allows for the inclusion of higher-order relationships in GLMs. This extension is beneficial when linear models prove insufficient for capturing the intricacies of the data.
General Functional Scope: Polynomials serve as standalone tools for approximating functions across various modeling scenarios. By introducing polynomial terms, you can capture non-linear trends independently of GLM structures. The degree of the polynomial determines the flexibility in modeling complex relationships. When used alone, polynomials are particularly useful for representing functions that exhibit curvature and intricate patterns.

## Splines:

Scope within GLM: Splines are piecewise-defined polynomials that allow for smoother transitions between different segments of a function. In the GLM context, splines are valuable for capturing local variations in relationships.
General Functional Scope: Splines, being piecewise-defined smooth functions, find broad applicability as standalone tools for functional approximation. Whether modeling curves or surfaces, splines allow for the representation of intricate, non-linear patterns. Their ability to smoothly connect different segments makes them suitable for capturing local variations independently of GLMs. Splines provide a flexible approach for approximating functions with varying degrees of smoothness.

## Piecewise Linear Models (e.g., Gaussian Mixtures):

Employing piecewise linear models, such as Gaussian mixtures, as standalone tools allows for the decomposition of complex relationships into distinct linear components. Each segment, modeled by a linear function, captures specific characteristics of the data. When used independently, piecewise linear models are effective for approximating functions with localized complexities and diverse linear patterns.

In each case, the objective is to employ these functional approximations independently, not necessarily within the GLM framework, with the understanding that MLPs can subsequently be introduced for further refinement and adaptation to the data. The combination of these versatile functional approximators with MLPs enhances the model's ability to capture intricate relationships and non-linearities across diverse modeling contexts.


# References

(1) [Zheng, Xun, Bryon Aragam, Pradeep K. Ravikumar, and Eric P. Xing. "Dags with no tears: Continuous optimization for structure learning." Advances in neural information processing systems 31 (2018).](https://proceedings.neurips.cc/paper_files/paper/2018/file/e347c51419ffb23ca3fd5050202f9c3d-Paper.pdf)

(2) [Zheng, Xun , Chen Dan , Bryon Aragam, ](https://arxiv.org/pdf/1909.13189.pdf)[Pradeep K. Ravikumar](https://proceedings.neurips.cc/paper_files/paper/2018/file/e347c51419ffb23ca3fd5050202f9c3d-Paper.pdf) , and Eric P. Xing :" Learning Sparse Nonparametric DAGs" Proceedings of Machine Learning Research [ (2020)](https://arxiv.org/pdf/1909.13189.pdf)