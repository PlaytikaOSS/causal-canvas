<!-- This page was ported from the GitHub wiki. -->

# Introduction

Bayesian networks, also known as probabilistic graphical models, are a powerful framework for modeling and reasoning under uncertainty. They provide a graphical representation of probabilistic relationships between variables and enable efficient inference and decision-making.

The key concept behind Bayesian networks is the conditional probability. A Bayesian network consists of two components: a directed acyclic graph (DAG) and a set of conditional probability tables (CPTs). The DAG represents the dependencies among variables (the component we get from Structural Learning), where each node in the graph represents a random variable, and the directed edges denote the probabilistic relationships between the variables. They enable probabilistic reasoning by incorporating evidence and updating probabilities accordingly. Once evidence is observed for a particular variable, it can be incorporated by modifying the conditional probabilities of the affected variables in the network. Bayesian networks use Bayes' theorem to update probabilities based on new evidence. This allows for updating beliefs as more information becomes available.

![A simple example of a DAG](https://github.com/PlaytikaOSS/causal-canvas/raw/main/images/simple_dag.png?raw=true)

**Figure 1:** A simple example of a DAG.

Assume the DAG in Figure 1. For each node X_i, a conditional probability p(x_i|pa(x_i)) is defined, where pa(x_i) denotes the parents of a random variable i, i.e., the set of features that have a direct regulatory influence on that ith feature. The set of random variables is completely determined by the joint probability distribution. Under the Markov assumption, i.e., the assumption that each feature is conditionally independent of its non-descendants given its parents, this joint probability distribution can be determined by the factorization via 

![Diagram: eq 3 1](https://github.com/PlaytikaOSS/causal-canvas/raw/main/images/eq_3_1.png)

## Conditional Probability Tables (CPTs)

Conditional Probability Tables (CPTs) are an essential component of Bayesian networks. They quantify the probabilistic relationships between variables by specifying the conditional probabilities of a node given its parents in the network.

Each node in a Bayesian network has its own CPT associated with it. The CPT of a node lists the probabilities of all possible states of that node, conditioned on each possible combination of states of its parents. The CPT provides a way to express how the probability of a particular node's state changes based on the states of its parents.

Let's take an example to illustrate the concept of CPTs:

Suppose we have the Bayesian network above and we want to see only the relationship between two variables: `churn` and `is_active`. The `churn` (target) variable has two states, "Yes" and "No," while the `is_active` variable also has two states, "Yes" and "No." The `churn` variable is directly influenced by the `is_active` variable.

The CPT for the "churn" node would look like this:

| **is_active** | **churn==Yes** | **churn==No** |
|---------------|----------------|---------------|
| Yes           |     0.2        |     0.8       |
| No            |     0.9        |     0.1       |


In this table, the rows correspond to the possible states of the `is_active` variable (Yes/No), and the columns correspond to the possible states of the `churn` variable (Yes/No).

Reading the CPT, we can interpret the probabilities as follows:

If the user is active, the probability of the user to churn is 0.2 and the probability of not churning is 0.8.
If it is not active, the probability to churn is 0.9, and the not churning is 0.1.

CPTs capture the conditional dependencies between variables in the Bayesian network. They provide a way to encode prior knowledge or expert judgment about the relationships between variables in a quantitative form. These probabilities can be estimated from data or elicited from domain experts.

The CPTs play a crucial role in inference and reasoning in Bayesian networks. They are used to compute probabilities, update beliefs, and make predictions based on the evidence observed in the network. Through the manipulation and combination of probabilities in the CPTs, the Bayesian network can propagate information and quantify the uncertainty in the variables of interest.

It's worth noting that CPTs can become large and complex in larger networks with many variables. In such cases, various techniques, such as parameter learning from data or using expert knowledge, can be employed to estimate or specify the probabilities accurately.

## Discretisation

Discretisation of numerical features is often necessary when working with Bayesian networks since they rely on discrete variables and the CPTs, which require discrete states. Discretisation is the process of transforming continuous numerical features into a set of discrete intervals or categories. Furthermore, discretisation simplifies the modeling process by reducing the complexity of the network. Working with discrete variables and discrete intervals in CPTs is computationally more efficient than dealing with continuous values, while it also helps in handling uncertainty and capturing dependencies between variables. By dividing the numerical feature into intervals, we can encode the relationships between variables using the conditional probabilities associated with each interval.

Now, let's explore three common techniques used for discretising numerical features in Bayesian networks:

### Simple Quantile Discretisation

Simple quantile discretisation divides the range of values into equal-sized intervals based on the quantiles of the data. The process involves sorting the numerical values and dividing them into intervals such that each interval contains an equal number of observations or an equal range of values. For example, if we have 100 data points and want to discretise into 5 intervals, each interval would contain 20 data points or an equal range of values. Simple quantile discretisation ensures that the intervals capture the distribution of the data and maintain an equal representation across intervals.

### Decision Tree Discretisation

Decision tree discretisation involves using a decision tree algorithm to partition the data into intervals based on the splitting criteria of the tree. The decision tree algorithm recursively finds the best split points in the numerical feature to maximise the information gain or another relevant criterion. The splitting points determined by the decision tree algorithm form the boundaries of the intervals for discretisation. Decision tree discretisation adapts to the data distribution and can capture nonlinear relationships between the feature and the target variable.

### Multi-Interval Discretisation (MDL)

Multi-Interval Discretisation is a technique that seeks to find the optimal set of intervals for discretising a numerical feature. It uses the Minimum Description Length (MDL) principle to minimise the description length of the data given the discretised representation. The method finds the intervals that maximise the compression of the data while minimising the loss of information; it balances the complexity of the model with the accuracy of representation and avoids overfitting.

## Parameter Estimation

There are three approaches commonly used to estimate the parameters of the CPTs from data: Maximum Likelihood Estimation (MLE), and Bayesian estimation under either a K2 prior or a BDeu prior. MLE picks the parameters that maximise the data likelihood; the Bayesian approaches treat each conditional distribution as a Dirichlet posterior whose pseudocounts encode prior knowledge. (Network *structure* learning is a separate problem covered elsewhere.)

Let's consider a Bayesian network with a node X, which has a set of parents denoted as pa(X). We denote the instantiation of the parents as P(pa(X)), where P(pa(X)) represents a specific combination of values for the parents.

The distribution of the node X, conditioned on its parents' instantiation, can be written as a set of Multinomial distributions. Let's denote the distribution of X as P(X | pa(X)).

The Multinomial distribution is parameterised by a set of probabilities, where each probability represents the likelihood of X taking on a specific state or value. In the context of the Bayesian network, the Multinomial distribution captures the conditional probabilities of X given its parents.

Mathematically, we can express the distribution of X as follows:


P(X | pa(X)) = { p(X=x | P(Parents(X))) : for all possible states x of X }, i.e.,

![Diagram: eq 3 2](https://github.com/PlaytikaOSS/causal-canvas/raw/main/images/eq_3_2.png)

Here, p(X=x | P(pa(X))) represents the probability of X taking on the specific state x, given the instantiation P(Parents(X)) of its parents. 

Each P(X_i | pa(X_i)) is translated as a conditional probability table (CPT), thus multiplying all the CPTs results in a joint distribution over all variables. Additionally, each P(X_i | pa(X_i)) means that a variable is independent of all its non-effects (non-descendants) given its direct causes (i.e. parents).


The set of Multinomial distributions in this expression represents the probabilities associated with each state of X, conditioned on every possible combination of values for its parents. Each instantiation of the parents corresponds to a specific Multinomial distribution that captures the conditional probabilities of X given that instantiation.

Note that the specific form of the Multinomial distributions and the calculation of the probabilities depend on the nature of the variables and the details of the Bayesian network. However, the general concept is that each node in the network has a set of Multinomial distributions, one for each instantiation of its parents, which capture the conditional probabilities of the node given its parents.

Now assume we have p variables X_1,…, X_p, where each X_i has r_i different categories and q_i denotes the number of distinct configurations of its parents (i.e. q_i = ∏_{X_j ∈ pa(X_i)} r_j).

We need to estimate each probability θ_ijk = P(X_i = j | pa(X_i) = k), where i = 1, …, p, j = 1, …, r_i, and k = 1, …, q_i; equivalently, the full parameter vector θ.

As each node X is a set of multinomial distributions one for each instantiation of its parents there are 3 approaches we can explore for parameter inference.

### MLE approach

Maximise likelihood function L(θ|Χ) to find parameters  where

![Diagram: eq 3 3](https://github.com/PlaytikaOSS/causal-canvas/raw/main/images/eq_3_3.png)

### Bayesian Estimation approach

In a simple X→ Y example with a given dataset D with M dimension of data, we want to compute the probability of instance  X_{M+1}, Y_{M+1} given X_{1}, Y_{1}, ..., X_{M}, Y_{M}. In every Bayesian setting, the predictive distribution means integration over the parameter vector, i.e.,

![Diagram: eq 3 4](https://github.com/PlaytikaOSS/causal-canvas/raw/main/images/eq_3_4.png)

Since the probability distribution decomposes into two products, i.e.,  

![Diagram: eq 3 5](https://github.com/PlaytikaOSS/causal-canvas/raw/main/images/eq_3_5.png)

then the prediction problems for X and Y can be solved separately.

#### Priors

Each of these θ_ijk has a separate Dirichlet prior governed by hyperparameters ![Diagram: eq 3 6](https://github.com/PlaytikaOSS/causal-canvas/raw/main/images/eq_3_6.png) which are called the "pseudocounts". 

Assuming that r_i is the number of total values (categories) that X_i has, and

![Diagram: eq 3 7](https://github.com/PlaytikaOSS/causal-canvas/raw/main/images/eq_3_7.png)

Then each probability of instance can be predicted.

##### K2 prior approach

The K2 prior is a Dirichlet prior over the CPT parameters in which all pseudocounts are set to one: α_ijk = 1 for every i, j, k. This is equivalent to Laplace smoothing on each conditional distribution and corresponds to a uniform prior over the parameter simplex.

The pseudocounts are fixed at one regardless of the cardinality of the variable or the number of its parent configurations:

![Diagram: eq 3 8](https://github.com/PlaytikaOSS/causal-canvas/raw/main/images/eq_3_8.png)

##### BDeu approach

We fix an imaginary sample size α and a distribution P'[X_1,…,X_p] of event frequencies:
![Diagram: eq 3 9](https://github.com/PlaytikaOSS/causal-canvas/raw/main/images/eq_3_9.png)

and this will divide data set proportionately. 

Number of imaginary samples for different choices of parents will be identical.

![Diagram: eq 3 10](https://github.com/PlaytikaOSS/causal-canvas/raw/main/images/eq_3_10.png)

This approach satisfies the score equivalence property which is very important in structural learning.

## Resources

- Srihari, *Bayesian Estimation of Bayesian Network Parameters* — <https://cedar.buffalo.edu/~srihari/CSE674/Chap17/17.2-BayesianBNParams.pdf>
- *Prior and Posterior Dirichlet Distributions on Bayesian Networks* — <https://pubs.aip.org/aip/acp/article/1827/1/020036/1020356/Prior-and-posterior-dirichlet-distributions-on>
- Cooper & Herskovits (1992), *A Bayesian Method for the Induction of Probabilistic Networks from Data* — <https://link.springer.com/article/10.1007/BF00994016>