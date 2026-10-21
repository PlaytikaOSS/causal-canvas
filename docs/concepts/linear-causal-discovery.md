<!-- This page was ported from the GitHub wiki. -->

# Linear Causal Discovery
![Diagram: why no tears](https://github.com/PlaytikaOSS/causal-canvas/raw/main/images/why_no_tears.png)
Graph taken from (2).

## Structural Equation Models

For a pair of variables Χ and Υ we can estimate if and how X→Y with:
![Diagram: noise](https://github.com/PlaytikaOSS/causal-canvas/raw/main/images/noise.png)

For now, we will focus on the most simplified version of Structural Equation Models, that of Linear noise.

In a linear additive noise setting, structural equation models (SEMs) provide a framework for analysing the relationships between variables by modeling both the direct and indirect effects among them. SEMs assume that the relationships between variables can be represented as a system of linear equations with additive noise.

The basic idea behind SEMs is to represent each variable as a latent (unobserved) variable that is influenced by other latent variables and measured by observable variables. The relationships between latent variables are expressed as structural equations, which are linear equations representing the direct effects. The relationships between the latent variables and the observed variables are represented as measurement equations.

In a linear additive noise setting, the structural equation is typically expressed as:

Y = B * X + E

where Y is the vector of observed variables, X is the vector of latent variables, B is the matrix of coefficients representing the direct effects of the latent variables on the observed variables, and E is the vector of error terms representing the additive noise.

The measurement equations in SEMs specify how the latent variables are measured by the observed variables. These equations typically take the form of:

Y = L * X + ζ

where Y is the vector of observed variables, X is the vector of latent variables, L is the matrix of coefficients representing the relationship between the latent variables and the observed variables, and ζ is the vector of measurement errors.

By estimating the coefficients in the structural and measurement equations, SEMs allow for the examination of both the direct and indirect effects of variables on each other. The estimation process involves minimizing the discrepancy between the observed data and the model-implied covariance matrix, typically using maximum likelihood estimation or other more advanced estimation methods.

## Optimisation of SEMs with NOTEARS

### NOTEARS

The NOTEARS (Non-combinatorial Optimization via Trace Exponential and Augmented lagRangian for Structure learning) (1) algorithm is a method for discovering causal relationships in a data set. It is specifically designed to learn the structure of Directed Acyclic Graphs (DAGs) from observational data. The algorithm is based on the assumption that the underlying causal relationships among variables can be represented by a DAG.

Learning DAGs from data is a difficult problem because of the combinatorial acyclicity constraint, which is hard to enforce efficiently. The paper introduces a new approach called score-based learning of DAGs. It transforms the traditional combinatorial optimsation problem into a continuous one. The objective is to minimise a score function, subject to constraints that ensure the resulting graph is acyclic. By formulating the problem as a continuous one, the paper eliminates the need for specialised algorithms that search through the combinatorial space of DAGs. Instead, standard numerical algorithms for constrained problems can be used , such that L-BFGS, making the implementation easier. This approach is similar to the advancements made in learning undirected graphical models using continuous log-det programs.


![Diagram: combinatorial to smooth](https://github.com/PlaytikaOSS/causal-canvas/raw/main/images/combinatorial_to_smooth.png)
Figure 1: From a combinatorial to a smooth problem

As seen in Figure 1, the authors explicitly construct a smooth function with computable derivatives that encodes the acyclicity constraint. This smooth function allows them to replace the combinatorial constraint with a smooth equality constraint, making the optimisation problem more amenable to standard numerical solvers. Furthermore, by considering an equality-constraint, then we can have a simultaneous estimation of the structure and parameters of a sparse DAG from high-dimensional data. 

![Diagram: NOTEARS algo](https://github.com/PlaytikaOSS/causal-canvas/raw/main/images/NOTEARS_algo.png)
Figure 2: NOTEARS for discovery


### NOTEARS for SEMs

Let X defined in R^nxd be a data matrix consisting of n i.i.d. observations of the random vector X = (X_1,...,X_d) and let D denote the (discrete) space of DAGs G = (V, E) on d nodes. Given X, we seek to learn a DAG G in D (also called a Bayesian network) for the joint distribution P(X) [22, 42]. X is modeled via a structural equation model (SEM) defined by a weighted adjacency matrix W in R^dxd. Thus, instead of operating on the discrete space D, we will operate on the continuous space of d x d real matrices.

Any W defined in R^dxd defines a graph on d nodes in the following way: Let A(W) in {0, 1}^dxd be the binary matrix such that [A(W)]ij = 1 <=> w_ij not equal to zero and zero otherwise; then A(W) defines the adjacency matrix of a directed graph G(W). In addition to the graph G(W), W = [ w_1 | ··· | w_d ] defines a linear SEM by X_j = w^{T}_{j} X + z_j , where X = (X_1,...,X_d) is a random vector and z = (z_1,...,z_d) is a random noise vector. More generally, we can model X_j via a generalized linear model (GLM) E(X_j | Xpa(X_j )) = f(w^{T}_{j} X). For simplification, we focus on linear SEM and the least-squares loss

![Diagram: eq 1](https://github.com/PlaytikaOSS/causal-canvas/raw/main/images/eq_1.png) (1)

although everything in the sequel applies to any smooth loss function defined over Rdxd. 

The 2 statistical properties of the LS loss (1) in scoring DAGs have been extensively studied:

The minimizer of the LS loss provably recovers a true DAG with high probability on finite-samples and in highdimensions (d n), and hence is consistent for both Gaussian SEM [3, 45] and non-Gaussian SEM [24].
Note also that these results imply that the faithfulness assumption is not required in this set-up.

Furthermore, since we are interested in learning a sparse DAG, the authors add L1 regularisation such that

![Diagram: eq 2](https://github.com/PlaytikaOSS/causal-canvas/raw/main/images/eq_2.png) (2).

Combining equations (1) and (2) we seek to solve the equation

![Diagram: eq 3](https://github.com/PlaytikaOSS/causal-canvas/raw/main/images/eq_3.png) (3)


As equation (3) is a smooth function and Theorem 1 bellow, the right part of Figure 1 is met

![Diagram: theorem 1](https://github.com/PlaytikaOSS/causal-canvas/raw/main/images/theorem_1.png)
 and the problem is defined as an equality constrained program for equation 3.

![Diagram: ECP](https://github.com/PlaytikaOSS/causal-canvas/raw/main/images/ECP.png)

The way that the authors solve the ECP above is with the augmented Lagrangian method [e.g. 25]. More details on the proof can be found in the NeurIps paper.

### Thresholding

It is known that post-processing estimates of coefficients through hard thresholding can effectively reduce the number of false discoveries, as demonstrated in previous studies [46, 52].

Motivated by these positive findings, the authors propose a similar approach for thresholding the edge weights in their algorithm. After obtaining a stationary point W_ECP of the optimization problem, they apply a fixed threshold value. Any edge weights in W_ECP that have an absolute value smaller than the threshold are set to zero. This thresholding strategy serves two purposes.

It helps to "round" the numerical solution of the augmented Lagrangian used in their optimization process. Due to numerical precision limitations, the solution W_ECP may not strictly satisfy the equality constraint h(W_ECP) = 0, but rather h(W_ECP) ≤ ε, where ε is a small tolerance near the machine precision (e.g., ε = 10^(-8)). By applying the thresholding step, the solution is effectively rounded to a value closer to zero, improving its adherence to the acyclicity constraint.
Since h(W_ECP) explicitly measures the "DAG-ness" of the solution W_ECP, setting a small threshold value is sufficient to exclude edges that could potentially induce cycles in the graph. In other words, the thresholding step helps eliminate edges that are less likely to be part of a directed acyclic graph (DAG).
Summary of process

Here is a simplified explanation of how the NOTEARS algorithm works:

- **Preprocessing**: We start by standardising the variables and determining their pairwise empirical covariance matrix.

- **Initialisation**: Random initialisation of a matrix called the "adjacency matrix" that represents the candidate edges in the DAG. Each element of the matrix represents the presence or absence of an edge between two variables.

- **Iterative Update**: Iteratively updates the adjacency matrix to minimise the loss function in equation 3. This loss function penalises deviations from the assumed linear additive noise model and encourages the estimation of a sparse DAG.

- **Soft Thresholding**: After each update, the adjacency matrix undergoes a soft thresholding operation. Soft thresholding sets small values in the matrix to zero while preserving the sign and magnitude of larger values. This step promotes sparsity and helps the algorithm identify the significant causal relationships.

- **Convergence**: The iterative update and soft thresholding steps are repeated until the algorithm converges. Convergence occurs when the adjacency matrix no longer changes significantly between iterations.

- **Directed Acyclic Graph**: The final adjacency matrix obtained from the algorithm represents the structure of the estimated DAG. It indicates the presence or absence of causal edges between variables, providing insights into the causal relationships among them.

The NOTEARS algorithm is known for its ability to handle nonlinear relationships and identify causal structures even when the relationships are complex. It is based on principles from optimisation theory and graphical models, and it leverages the sparsity assumption to discover meaningful causal relationships.

It's worth noting that the original NOTEARS algorithm is designed for continuous variables and assumes a linear additive noise model. However, extensions and adaptations of the algorithm have been proposed to handle discrete variables and nonlinear relationships in causal discovery.

### Extensions of NOTEARS

**DYNOTEARS**: Time series extension using Causal Vector Autoregressive Models in (3).

**NOBEARS**: Non linear extension of NOTEARS in (4).

**NOFEARS**: More scalable extension of NOTEARS in (5)

#### Benefits of using NOTEARS
- Causal Structure Discovery: It is specifically designed for causal structure discovery, which means it is tailored to identifying the causal relationships among variables in a dataset, whereas other Deep Learning approaches for structure discovery, including node classification and link prediction, may not inherently prioritise causal discovery.

- Causal Assumptions: It is built on the principle of the "faithfulness assumption," which assumes that the data is generated from a causal model that adheres to certain statistical independence constraints. This assumption aligns well with the goals of causal inference and thus are explicitly designed for causal discovery.

- Scalability: NOTEARS and related causal discovery algorithms can be efficient and scalable, making them suitable for large datasets whereas depending on their architecture and complexity, other architectures such us GNNs may require more computational resources and time for training.

- Causal Constraints: NOTEARS allows you to impose causal constraints and prior knowledge on the structure learning process. You can guide the algorithm with domain expertise, which can be crucial in certain applications. Other approaches do not inherently provide mechanisms for incorporating such constraints.

- Data Efficiency: It can work with relatively small datasets, especially when combined with domain knowledge or expert insights. 
- Ranking-Based Stability: It employs a ranking-based approach to assess the stability of the learned structures across different datasets or subsamples. This stability criterion helps in identifying robust and reliable structures, reducing the risk of overfitting and providing a degree of uncertainty estimation that is lacking in some other methods.

- Theoretical Guarantees: It offers theoretical guarantees related to consistency and identifiability, which means that under certain conditions, it can accurately recover the true underlying graphical model structure. These guarantees provide a solid foundation for its use in practical applications.

- No Arbitrary Parameter Tuning: Unlike some other methods that require manual tuning of parameters, NOTEARS has minimal hyperparameters and typically does not require extensive user intervention. This simplifies its use and makes it accessible to researchers and practitioners without extensive domain knowledge.

- Empirical Performance: NOTEARS has demonstrated impressive empirical performance in benchmark datasets and real-world applications. It often outperforms or matches the accuracy of other state-of-the-art algorithms, showcasing its effectiveness in practical scenarios.

#### Other notable approaches

**PC Algorithm (Peter and Clark Algorithm)**:

The PC algorithm is a widely-used and classic method for learning the structure of Bayesian networks. It employs conditional independence tests to identify and remove edges from an initially fully connected graph. This iterative process gradually refines the network structure until a Markov equivalence class is obtained.

**Grow-Shrink Algorithm**:

The Grow-Shrink algorithm is another approach for learning the structure of Bayesian networks. It starts with an empty graph and iteratively adds and removes edges based on conditional independence tests. This method aims to discover the true underlying network structure while avoiding overfitting.

**GES (Greedy Equivalence Search)**:

GES is a score-based algorithm for learning equivalence classes of Bayesian network structures. It combines both forward and backward passes to identify the Markov equivalence class that best fits the data. GES is known for its ability to handle latent variables and non-Gaussian data.

**MMHC (Max-Min Hill-Climbing)**:

MMHC is a hybrid algorithm that combines the strengths of the hill-climbing and MMHC phases. It performs local search using a hill-climbing approach and utilizes a hybrid scoring criterion to select the best structure. MMHC is efficient and robust in various data scenarios.

**Fast Causal Inference (FCI)**:

FCI is an algorithm designed to discover the causal relationships among variables in observational data. It incorporates conditional independence tests and orientation rules to identify the Markov equivalence class of causal graphs. FCI is especially useful when dealing with causal inference tasks.

**Bayesian Structure Learning**:

Bayesian methods for structure learning use Bayesian scoring criteria such as the Bayesian Information Criterion (BIC) or Bayesian Dirichlet Equivalent (BDe) score to infer the network structure. These methods provide a probabilistic framework for model selection, incorporating prior beliefs about the network structure.

**Hybrid Approaches**:

Some recent approaches combine elements of different algorithms, such as integrating PC and hill-climbing techniques. These hybrid methods aim to leverage the strengths of multiple algorithms for more robust and accurate structure learning.

### Resources

(1) [Zheng, Xun, Bryon Aragam, Pradeep K. Ravikumar, and Eric P. Xing. "Dags with no tears: Continuous optimization for structure learning." Advances in neural information processing systems 31 (2018).](https://proceedings.neurips.cc/paper_files/paper/2018/file/e347c51419ffb23ca3fd5050202f9c3d-Paper.pdf)

(2) [Upadhyaya P, Zhang K, Li C, Jiang X, Kim Y. Scalable Causal Structure Learning: Scoping Review of Traditional and Deep Learning Algorithms and New Opportunities](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9890349/)

(3) https://arxiv.org/abs/2002.00498

(4) https://psb.stanford.edu/psb-online/proceedings/psb20/Lee.pdf

(5) https://arxiv.org/abs/2010.09133