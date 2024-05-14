# Causal Canvas

This repository provides a tool for Causal discovery with Structural Learning and probabilistic modelling of
the learnt structure.
It uses `causalnex` library in which a fully connected graph is initial assumed
but then the structure is learned through the NOTEARS algorithm by [Zheng et al. (2018)](https://proceedings.neurips.cc/paper_files/paper/2018/file/e347c51419ffb23ca3fd5050202f9c3d-Paper.pdf).

New addition: Non-linear Structure Learning with NOTEARS-MLP, please navigate to our dedicated [wiki page](https://wiki.playtika.com/display/AILAB/1.2+Non-linear+Causal+Discovery) for more information.

More information on `causalnex` library can be found in the [official python documentation](https://causalnex.readthedocs.io/en/latest/index.html#).

The tool conducts as well the do why calculus of Judea Pearl.

This script performs the following steps for now:

* Read input configuration file
* Load data
* Preprocess data
* Fit structure learner (Linear NOTEARS or non-linear NOTEARS)
* Calculate fully connected graph with constraints that event, e.g. churn cannot cause features
* Calculate largest subgraph
* Calculate pruned graph based on threshold `w`
* Calculate largest subgraph from the pruned graph
* Calculate pruning up to the largest DAG
* Discretise numerical features
* Fit the Bayesian Network's probabilistic model
* Extract predicted conditional probabilities of a node within segments wrt its parents
* Save results in the output directory
* Conduct error analysis on multiple models

For a comprehensive explanation on Causal Discovery, Bayesian Network and Do-why inference within the scope of this
tool please direct in:

* [1. Causal Canvas Introduction](https://github.com/Playtika/causal_canvas/wiki/1.-Causal-Canvas-Introduction)
* [2. Linear Causal Discovery](https://github.com/Playtika/causal_canvas/wiki/2.-Linear-Causal-Discovery)
* [3. Non‐linear Causal Discovery](https://github.com/Playtika/causal_canvas/wiki/3.-Non%E2%80%90linear-Causal-Discovery)
* [4. Bayesian Networks Estimation](https://github.com/Playtika/causal_canvas/wiki/4.-Bayesian-Networks-Estimation)
* [5. Interventions and counterfactuals for decision making](https://github.com/Playtika/causal_canvas/wiki/5.-Interventions-and-counterfactuals-for-decision-making)

## Code structure

Source code

```text
├── docs                                                  <- Documentation.
│   │   └── config.yaml                                   <- Dummy experiment configuration.
├── causal_canvas                                         <- Directory of the source code.
│   ├── bayesian_network_estimator.py                     <- Bayesian Network fitting class.
│   ├── custom_no_tears_mlp.py                            <- Class of Non-linear causal discovery with MLP.
│   ├── inference_utils.py                                <- Utilities functions for do why inference.
│   ├── evaluation_utils.py                               <- Utilities functions for error analysis.
│   ├── nonlinear_notears.py                              <- Utilities functions for running non-linear discovery.
│   ├── preprocessor.py                                   <- Data preprocessing code.
│   ├── script_config.py                                  <- Configurator.
│   ├── script_config_evaluation.py                       <- Configurator for error analysis.
│   ├── script_config_inference.py                        <- Configurator for dowhy inference.
│   ├── structure_learner.py                              <- Causal discovery class with NO-TEARS algorithm
│   └── utils.py                                          <- Utilities functions.
├── evaluate_experiments.py                               <- Python script to execute for error analysis.
├── infer_bayesian_network.py                             <- Python script to execute for causal discovery and Bayesian Network fitting.
├── infer_do_why.py                                       <- Python script to execute for do why inference on known structure and bayesian network.
├── repeat_experiments.py                                 <- Python example script to execute for multiple model trials.
├── requirements.txt                                      <- Python libraries requirements for the script to run.
└── README.md                                             <- Info causal inference tool and how to use it.
```

## Causal Discovery and Bayesian Network Fitting

### Configuration file for Causal Discovery and modelling

The configuration file should respect the following structure.
Examples of config files are also at ```lausanne_player_360/scripts/causal_inferences/docs/config.yaml```.

* ```config.yaml```
  * ```validation_dates```: date range of validation set to split the data before preprocessing
  * ```data_input_path```: path of the input dataset, e.g. "path/data.parquet"
  * ```structure_path```: path of an already learnt structure, e.g. "path/DAG.dot". If not specified, it runs causal discovery and output that structure. If provided, discovery is not conducted and uses the given structure for Bayesian network inference.
  * ```id_column```: name indicating id column, e.g. user_id
  * ```date_column```: name indicating date column, e.g fs_calc_date
  * ```event_column```: name indicating event column, e.g churn
  * ```drop_columns```: columns to drop, e.g. ["age", "school"]
  * ```features_select```: features to be selected for modelling
  * ```categorical_features```: categorical features to be separated during data scaling
  * ```tabu_edges```: list of lists with directional causal effects that cannot happen. First position of sublist should
    point to F1 -> F2, i.e., ["G2", "Fedu"] indicates that `G2` cannot cause `Fedu`
  * ```tabu_edge_features```: lift of features that cannot be inflicted by the rest of features, e.g. `Fedu`
  * ```event_graph_label```: label to be shown for the node for target column, i.e., CHURN, WILL_SINK etc. Can type either colour
    is CSS or a name, e.g., `black` etc.
  * ```event_color```: colour of the node for target column
  * ```higher_contribution_feature_color```: colour of node for the feature with highest causal impact to target
  * ```invert_signs```: Boolean indicating whether the edges signs should be reverted visually for better interpretation. Default is False.
  * ```sample_frac```: float between 0 and 1 to perform the modelling and analysis on a sample of the data.
  * ```h_tol```: exit if h(W) < h_tol (as opposed to strict definition of 0) in NOTEARS
  * ```max_iter```: max number of dual ascent steps during optimisation in NOTEARS.
  * ```w_threshold```: fixed threshold for absolute edge weights to prune the graph in NOTEARS.
  * ```lasso_multiplier```: if selected, constant that multiplies the lasso term in Linear NOTEARS.
  * ```connections_type``` : whether to use "linear" or "non_linear" Causal Discovery.
  * ```non_linear_args```: If this dictionary arguments are passed, then they are considered to conduct non-linear Causal Discovery.
      -```feature_to_distribution_map```: Dictionary with mapping of a feature to a distribution. Distribution values can be
          "bin" for binary, "cat" for Multinomial/Categorical, "cont" for Gaussian, "pos_cont" for non-negative defined Gaussian,
    "bimodal_pos_cont": Mixture of two non-negative defined Gaussian (currently on testing), "ord" for Ordinal distribution,
    "poiss" for Poisson distribution, "gamma": for gamma distribution and "tweedie" for tweedie distribution, currently applied for scale factor = 1.5.
    * ```ridge_multiplier```: if selected, constant that multiplies the ridge term in MLP.
    * ```lasso_multiplier```: if selected, constant that multiplies the lasso term in MLP.
    * ```use_bias```: True to add the intercept to the model. Default is set to false.
    * ```hidden_layer_units```: An iterable where its length determine the number of layers used,  and the numbers determine the number of nodes used for the layer in order.
    * ```use_gpu```: use gpu if it is set to True and CUDA is available
    * ```verbose```: True to print the loss, h and rho during dual ascent steps.
  * ```discretiser```: dictionary with two arguments
    * ```method```: it can be either `simple`, i.e., simple quantile split discretisation), `tree`, i.e., Decision
    Trees where the cutting points on the Decision Tree become
    the chosen discretisation thresholds or `mdlp` algorithm (Fayyad and Irani, 1993) where dynamic split strategy based
    on binning the number of candidate splits is implemented to increase efficiency
    (Chickering, Meek and Rounthwaite, 2001).
    * ``argument``: for `simple` discretiser reflects the number of quantile splits for the
        features. For `tree` discretiser reflects the maximum tree depth. For `mdlp` discretiser is the minimum split value.
    * ``proportion_threshold``: for post-processing the segments that are created with discretisation. It should be a number
        between 0 and 1. Basically it will merge each category that is less than ``proportion_threshold`` with the previous one. Default is `None`.
    * ``max_categories``: number of maximum categories to have post discretisation. Default is set to `None`.
    * ``discretisation_cutoffs_target``: list of cutoffs to be used for a numeric target during bayesian network inference, i.e.,
     if value is given as `[0]` it will create a binary target from `(-inf, 10)`, `[10, inf)`, if value is given as `[0, 10]`
     it will create three categories etc.
  * ```inference_method```: dictionary with method and arguments for fitting the model.
    * ```method```: `MaximumLikelihoodEstimator` or `BayesianEstimator`
    * ```bayes_prior```: only if `BayesianEstimator` is chosen, it can be `K2` or `BDeu`
    * ```equivalent_sample_size```: used only with `BDeu` to tune the prior quantity
          `equivalent_sample_size / (node_cardinality * np.prod(parents_cardinalities))`
  * `conditional_dependency_estimates`: nodes for which we want to extract the conditional dependencies wrt to its parents. It can be `churn_label`, `is_active`, etc.
  * `create_new_folder`: whether it should create a new folder or rewrite
  * `add_datetime_to_folder`: whether to add datetime in folder. Useful if multiple experiments are being run.
  * ```output_path```: path of the output results dataset, e.g. "/mnt/VAST_SHARED/"

### Causal Discovery Usage

Create environment with conda:

```commandline
conda create --name <env name> python=3.10
conda activate <env name>
pip install -r requirements.txt
```

Create environment with poetry:

```commandline
poetry config virtualencs.in-ptoject True
poetry install
```

Run the following command line to execute the tool. Be careful to choose the right config file according to the specific task (either "regularised", "simple").

Run main scripts in conda environment:

```commandline
$ python3 infer_bayesian_network.py --config_file CONFIG_FILE
argument:
    --config_file CONFIG_FILE
                        configuration path of the specific process to run
```

e.g.

```commandline
python3 infer_bayesian_network.py --config_file path/config.yaml
```

Run main scripts with poetry:

```commandline
$ poetry run python infer_bayesian_network.py --config_file CONFIG_FILE
argument:
    --config_file CONFIG_FILE
                        configuration path of the specific process to run
```

e.g.

```commandline
poetry run python infer_bayesian_network.py --config_file path/config.yaml
```

Experiment results

```text
└── experiment                                                <- Directory for the experiment results.
    ├── causal_discovery                                      <- Directory for the causal discovery.
    │   ├── 01_fully_connected_graph.html                     <- HTML file with interactive fully connected graph with or without lasso.
    │   ├── 02_largest_subgraph.html                          <- HTML file with interactive largest subgraph.
    │   ├── 03_threshold_graph.html                           <- HTML file with prunned graph.
    │   ├── 04_threshold_DAG.html                             <- HTML file with resulting DAG after pruning.
    │   ├── 05_DAG_subgraph.html                              <- HTML file with largest subgraph of the prunned DAG.
    │   ├── label_encoder.pkl                                 <- Label encoder for categorical variables.
    │   ├── DAG_for_inference.dot                             <- Structured DAG networkx class to use for inference.
    │   ├── scaler.pkl                                        <- Scaler for numerical variables.
    │   ├── all_weights_mlp.csv                               <- If MLP is chosen, resulting weight matrix.
    │   ├── threshold_weights_mlp.csv                         <- If MLP is chosen, resulting weight matrix after hard threshold.
    │   ├── all_std_error_weights_mlp.csv                     <- If MLP is chosen, resulting s.e. of weight matrix
    │   ├── threshold_std_error_weights_mlp.csv               <- If MLP is chosen, resulting s.e. of weight matrix after hard threshold.
    │   ├── all_weights_heatmap.png                           <- If MLP is chosen, heatmap weight matrix and target weights after hard threshold.
    │   ├── all_weights_w_threshold_heatmap.png               <- If MLP is chosen, heatmap of weight matrix and target weights after hard threshold.
    │   ├── score_edges_full_structure.yaml                   <- Weights of structural causal relationships for the full sctructure with or without lasso.
    │   └── score_edges_sparser_structure.yaml                <- Weights of structural causal relationships for the sparser sctructure with or without lasso.
    ├── bayesian_network                                      <- Directory for the bayesian_network probabilistic modelling.
    │   ├── dependency_estimates                              <- Directory with node specific CPD wrt to its parents
    │   │   ├── node_name.csv                                 <- CSV with node specific CPD wrt to its parents.
    │   │        ...
    │   ├── model_fit                                     <- Directory bayesian_network model class and discretiser.
    │   │   ├── bayesian_network.pkl                      <- Bayesian Network model class
    │   │   ├── quantile_splits.json                      <- Dictionary with numerical splits by feature as decided by the discretiser
    │   │   ├── discretiser.pkl                           <- Discretiser model class if `tree` or `mdlp` is used
    │   │   └── condenced_splits.json                     <- Dictionary with numerical splits by feature as decided by post-processing with condensing categories
    └── config.yaml                                       <- Configuration file.

```

## Error analysis

In this phase, we perform an evaluation of various model combinations following Bayesian network inference.
Initially, we generate bootstrap intervals for F1-score, precision, and recall using a default threshold
of 50% for the optimisation dataset. Subsequently, we determine an optimized threshold based on the optimisation
dataset's results. Then, we recompute these performance scores for the test dataset.
Overall, this phase facilitates a clear organization of error analysis results
making it easy to access and interpret the performance of different BN models and inference methods at each step of the analysis process.
It allows for a comprehensive evaluation of model performance and the impact of threshold optimisation on the test set predictions.

This phase outputs four different folders:

1. *cpd_plots Directory*: This directory contains plots depicting the variation the Conditional Probability Tables (CPTs)
for different models. It provides insight into how variable the CPTs are in each segment.

2. *optim_set_predictions Directory*: Within this directory, you have subdirectories for different inference methods.
Each subdirectory contains predictions made for the optimisation set before threshold optimisation.
The predictions are stored in CSV files. Additionally, there's a CSV file and
a plot with bootstrap estimated scores for all models in this directory, providing an overview of model performance.

3. *test_set_predictions Directory*: Similar to the `optim_set_predictions` directory,
this one also contains subdirectories for different inference methods.
Inside each subdirectory, you'll find predictions made for the test set using
the optimized threshold obtained from the optimization set.
These predictions are also stored in CSV files. Like before, there's a CSV file and a plot with bootstrap estimated
scores for all models, allowing you to assess model performance on the test set.

The configuration file should respect the following structure.
Example of config file ```lausanne_player_360/scripts/causal_inferences/docs/config_evaluation.yaml```.

* ```config_evaluation.yaml```
  * ```data_train_input_path```: list of path(s) of the input dataset(s), e.g. "path/data.parquet". It can work for datasets that share the same preffix path, i.e., for pre-segmented datasets.
  * ```data_test_input_path```: list of path(s) of the test input dataset(s), e.g. "path/data.parquet". It can work for datasets that share the same preffix path, i.e., for pre-segmented datasets.
  If not specified, the it is assumed that test data is integrated with train data and optim_test_dates and test_dates need to be provided.
  * ```event_column```:
    * ```name``` indicating event column, e.g. `churn`, `G1`
    * ```target_type```: if `cont` then it doesn't do any categorical transformations prior to discretisation.
    * ```target_column_names_cpds```: list of bins created by bayesian network. They can be found via dependency_estimates
    extracted by Bayesian Network inference, e.g., `["[-inf, 10)", "[10, inf)"]`.
    * ```target_class```: "[10, inf)": desired target/minority class, i.e., if target is `churn` then we can specify `True`
    or `1`, if target is G3 from student data example, our target class is passing the exam, i.e., `"[10, inf)"]`.
  * ```date_column```: name indicating date column if needed, e.g `date`
  * ```optim_test_dates```: dates within the dataset to optimise the threshold on if needed, e.g `date`.
  * ```test_dates```: dates within the dataset to evaluate the model on after the threshold optimisation if needed, e.g `date`.
  * ```features_select```: features that were used for modelling
  * ```all_models_path```: prefix path of folder that includes all models, e.g., /mnt/VAST_SHARED/churn/all_models/
  * ```inference_methods```: list with all inference methods that the user wants to evaluate, i.e, [`mle`, `k2_prior`]. They represent the model subfolder within all_models_path, i.e. all_models_path/inference_method_path/
  * ```models_combinations```: dictionary including all the discretisation combinations that the user wants to evaluate. It represents the second downstream folder from models_path, i.e., `{'tree': [1, 2]}` maps to folders all_models_path/inference_method_path/tree_1 and all_models_path/inference_method_path/tree_2
  * ```score_to_optimise```: which score to optimise the threshold on. It can only be `f1-score`, `precision` or `recall`. If not specified then 50% threshold is used by default for the evaluation.
  * ```boot_iterations```: integer indicating the bootstrap iterations for the score calculations.
  * ```alpha```: the quantiles to extract with bootstrap.
  * ```use_multiprocessing```: if true, it uses multiprocessing to split the test sets in order to conduct the predictions.
  * ```output_path```: path of the output results dataset, e.g. "/mnt/VAST_SHARED/"

### Error Analysis Usage

Run the following command line to execute the error analysis tool.

```commandline
$ python3 evaluate_experiments.py --config_file CONFIG_FILE
argument:
    --config_file CONFIG_FILE
                        configuration path of the specific process to run
```

e.g.

```commandline
python3 evaluate_experiments.py --config_file path/config_evaluation.yaml
```

Error analysis results

```text
└── error_analysis                                            <- Directory for the error analysis results.
    ├── cpd_plots                                             <- Directory with plots comprising how variable the CPTs are in each segment.
    │   ├── cpds_mle_tree_2.png                               <- Plot of model specific CPTs.
    │   ├── cpds_mle_tree_4.png
    │   ├──  ...
    │   └── cpds_mle_tree_4.png
    ├── optim_set_predictions                                    <- Directory with all predictions that were conducted for the optim set before threshold optimisation with threshold = 50%.
    │   ├── inference_method_1                                   <- Directory with all predictions for a specific inference method that was used.
    │   │   ├── predictions_with_discretiser_tree_2.csv          <- CSV with predictions for the model trained with inference_method_1, discretiser and argument.
    │   │   ├── predictions_with_discretiser_mdlp_5.csv
    │   │        ...
    │   ├── inference_method_2
    │   ...
    │   ├── bootstrap_scores.csv                                <- CSV with bootstrap estimated scores for all models.
    │   ├── bootstrap_scores.png                                <- Plot with bootstrap estimated scores for all models.
    │   └── prediction_times.png                                <- Plot with time it took for each prediction.
    ├── test_set_predictions                                    <- Directory with all predictions that were conducted for the test set with using the optimised threshold taken from optim_set.
    │   ├── inference_method_1                                  <- Directory with all predictions for a specific inference method that was used.
    │   │   ├── predictions_with_discretiser_argument_1.csv     <- CSV with predictions for the model trained with inference_method_1, discretiser and argument.
    │   │   ├── predictions_with_discretiser_argument_2.csv
    │   │        ...
    │   ├── inference_method_2
    │   ...
    │   ├── bootstrap_scores.csv                                <- CSV with bootstrap estimated scores for all models.
    │   ├── bootstrap_scores.png                                <- Plot with bootstrap estimated scores for all models.
    │   └── prediction_times.png                                <- Plot with time it took for each prediction.
    └── config.yaml                                             <- Configuration file used for the error analysis.

```

## Do-why inference tool

This script performs Do-Why Bayesian Network inference based on a configuration file that defines various aspects
of the inference process. It is designed to analyze and understand causal relationships within a dataset using
probabilistic graphical models. The script leverages the CausalNex library for efficient Bayesian Network
manipulation and inference.

The script will perform Bayesian Network inference based on the provided configuration file,
analyzing causal relationships, calculating marginals, conducting interventions,
and generating various visualizations and output files.

For more details on the script's functionality, refer to the inline comments
and the detailed docstring provided within the script.

The configuration file should respect the following structure.
Examples of config files are also at ```lausanne_player_360/scripts/causal_inferences/docs/config_inference.yaml```.

* ```config_inference.yaml```
  * ```event_column```: name indicating event column, e.g churn.
  * ```target_class```: "[10, inf)": desired target/minority class, i.e., if target is `churn` then we can specify `True`
    or `1`, if target is G3 from student data example, our target class is passing the exam, i.e., `"[10, inf)"]`. The
    class specified will be the target class when estimating ATEs.
  * ```model_path```: path indicating where the BayesianNetwork class can be loaded.
  * ```conditionals```: dictionary to get conditional marginal distributions. i.e. `churn`: [`is_active`, `level_end`]
  translates to the conditional probabily distribution `P(churn|is_active, level_end)`, which means that any other direct
  effects towards churn are being marginalised out.
    * ```interventions```: dictionary mapped on a feature with probability shifts for each category. We can have assign multiple
  interventions, i.e., an intervention on `level_end` under the name `low_leveling` gives more probability/weight towards
  having more low level users in the game. Since the discretiser during inference in our example created three bins for
  `level_end` and all bins are ordered, we will put more probability on the first bin, i.e. `{0: 0.7, 1: 0.2, 2: 0.1}` etc.
  Or we can pick a single intervention such that with `impressed_popups_count` giving more probability towards less popus, i.e.
  `{0: 0.7, 1: 0.3}` is a `do_intervention('X', {'x': 0.7, 'y': 0.3})` will set :math:`P(X=x)` to 0.7, and :math:`P(X=y)` to 0.3.
  For now the tool resets the probabilities after each intervention. In the future we will extend
  the tool to incorporate combination of interventions, i.e. changing `level_end` and `impressed_popups_count` on the same
  time.
  * `counterfactuals`: list with features to calculate counterfactuals and ATE. `do_intervention('X', 'x')` will set :math:`P(X=x)` to 1, and :math:`P(X=y)` to 0
  * ```output_path```: path of the output results dataset, e.g. "/mnt/VAST_SHARED/"

## Author

Written by Sofia Maria Karadimitriou (<sofia.m.karadimitriou@gmail.com>)

## Citations

@software{Beaumont_CausalNex_2021,
author = {Beaumont, Paul and Horsburgh, Ben and Pilgerstorfer, Philip and Droth, Angel and Oentaryo, Richard and Ler, Steven and Nguyen, Hiep and Ferreira, Gabriel Azevedo and Patel, Zain and Leong, Wesley},
month = oct,
title = {{CausalNex}},
url = {<https://github.com/quantumblacklabs/causalnex}>,
year = {2021}
}
