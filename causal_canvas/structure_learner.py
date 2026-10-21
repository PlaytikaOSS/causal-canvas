"""
Structure learning entry point — NOTEARS-based DAG discovery.

Defines :class:`StructureLearner`, the high-level pydantic-validated
class that accepts a preprocessed DataFrame and returns a learned
``StructureModel`` (a ``networkx.DiGraph``) representing the causal DAG.
Handles tabu edges, knowledge-graph constraints, threshold pruning,
and persistence of the resulting structure.

**Layer**: high-level. First stage of the typical pipeline
(``Preprocessor`` → ``StructureLearner`` → ``BayesianNetworkEstimator``).

**Relationship to causalnex**: dispatches between
:mod:`causal_canvas.vendored.notears_linear` (linear NOTEARS, vendored from causalnex) and the local
:mod:`causal_canvas.nonlinear_notears` wrapper (MLP-based NOTEARS),
selected via the ``connections_type`` argument. The output type is
``StructureModel`` regardless of which backend ran,
so consumers see a uniform DAG interface.

See also
--------
:mod:`causal_canvas.nonlinear_notears` — non-linear MLP wrapper.
:mod:`causal_canvas.custom_notears_mlp` — low-level MLP solver.
:mod:`causal_canvas.bayesian_network_estimator` — primary downstream
    consumer of the learned DAG.
"""

import json
import pickle
from pathlib import Path
from typing import Annotated, Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from loguru import logger
from pydantic import BaseModel, Field

import causal_canvas.nonlinear_notears as nonlinear
import causal_canvas.vendored.notears_linear as linear
from causal_canvas.vendored.plots import plot_structure
from causal_canvas.vendored.structure_model import StructureModel


class StructureLearner(BaseModel):
    """
    Structure Learning with NOTEARS algorithm. Accepts a full dataset with features
    of interest, starts with a fully connected graph and the structure is learnt
    as a purely continuous optimization problem over real matrices that avoids this combinatorial
    constraint entirely and doesn't make any specific structural assumptions.

    Thresholding rules are used to avoid false discoveries (Zhou, 2009; Wang et al., 2016).
    Given a fixed threshold w > 0, the weights smaller than w are set in absolute value to zero.
    A small threshold w suffices to rule out cycle-inducing edges. An extra lasso multiplier can
    be considered in order to induce more regularisation in the algorithm.

    Zheng et al. (2018) https://arxiv.org/pdf/1803.01422.pdf

    Parameters
    ----------
    X: pd.DataFrame
        dataframe with numerical variables to conduct structure learning on
    event_col: str
        string indicating event column. It will create the tabu edges, i.e.,
        event cannot cause the features, but features are allowed to cause
        event.
    lasso_multiplier: float
        Constant that multiplies the lasso term.
    max_iter: int
        max number of dual ascent steps during optimisation
    h_tol: float
        exit if h(W) < h_tol (as opposed to strict definition of 0).
    w_threshold: float
        fixed threshold for absolute edge weights.
    tabu_edges: list
        list of edges(from, to) not to be included in the graph.
    tabu_edges: list
        list of features that are not inflicted by the rest of features,
        e.g. days_since_install
    event_label: str
        label to be shown for the node for target column, i.e., CHURN, WILL_SINK etc.
    event_color: str
        colour of the node for target column
    higher_contribution_feature_color: str
        colour of node for the feature with highest causal impact to target

    --- TBD
     - tabu_parent_nodes (Optional[List[str]]) – list of nodes banned from being a parent of any other nodes.
     - tabu_child_nodes (Optional[List[str]]) – list of nodes banned from being a child of any other nodes.
     - NORMAL, WEAK, STRONG NODES AND EDGES
     - extension for TS, DYNOTEARS
    """

    X: pd.DataFrame
    event_col: str
    lasso_multiplier: Optional[float] = None
    connections_type: Optional[str] = "linear"
    non_linear_args: dict
    max_iter: Annotated[int, Field(ge=1)]
    h_tol: Annotated[float, Field(gt=0)]
    w_threshold: Optional[float] = None
    tabu_edges: Optional[list] = None
    tabu_edge_features: Optional[list[str]] = None
    event_label: Optional[str] = None
    event_color: Optional[str] = "red"
    higher_contribution_feature_color: Optional[str] = "yellow"
    invert_signs: Optional[bool] = False
    structural_model: Optional[Any] = None
    threshold_structural_model: Optional[StructureModel] = None
    nonlinear_model: Optional[Any] = None

    model_config = {"arbitrary_types_allowed": True}

    def _require_structural_model(self) -> StructureModel:
        """Return the fitted structural_model or raise.

        Most methods on this class only make sense after ``learn_structure``
        (or ``discover_dag``) has run. This helper gives a clear runtime
        error instead of ``AttributeError: 'NoneType' object has no
        attribute ...``.
        """
        if self.structural_model is None:
            raise RuntimeError(
                "structural_model not set; call learn_structure or "
                "discover_dag before inspecting the learned graph"
            )
        return self.structural_model

    def _require_threshold_structural_model(self) -> StructureModel:
        """Return the fitted threshold_structural_model or raise.

        Only populated when ``w_threshold`` is truthy and learn_structure
        has run. Use at the top of code paths guarded by ``if
        self.w_threshold:``.
        """
        if self.threshold_structural_model is None:
            raise RuntimeError(
                "threshold_structural_model not set; either w_threshold "
                "is None or learn_structure has not been called"
            )
        return self.threshold_structural_model

    def _require_nonlinear_model(self) -> Any:
        """Return the fitted nonlinear_model or raise.

        Only populated for non-linear discovery (``connections_type !=
        'linear'``) after learn_structure.
        """
        if self.nonlinear_model is None:
            raise RuntimeError(
                "nonlinear_model not set; non-linear learn_structure has " "not been called"
            )
        return self.nonlinear_model

    def _get_all_tabu_edges(self):
        tabu_edges_with_target = []

        for feat_col in self.X.drop(self.event_col, axis=1).columns:
            tabu_edges_with_target.append((self.event_col, feat_col))

        if self.tabu_edge_features:
            for tab_feat in self.tabu_edge_features:
                for feat_col in self.X.drop(tab_feat, axis=1).columns:
                    tabu_edges_with_target.append((feat_col, tab_feat))

        if self.tabu_edges:
            tabu_edges_with_target = tabu_edges_with_target + [tuple(i) for i in self.tabu_edges]
        return tabu_edges_with_target

    def _learn_linear_structure(self):
        """
        Learns the structure of DAG via NOTEARS algorithm by assuming Linear SEMs.
        """
        logger.info("Initiating Causal Discovery with Linear SEMs")
        logger.info("Defining restricted edges")

        tabu_edges_with_target = self._get_all_tabu_edges()

        if not self.lasso_multiplier:
            logger.info(
                f"Structural learning with {self.max_iter} max iterations and tolerance of {self.h_tol}"
            )

            self.structural_model = linear.from_pandas(
                X=self.X,
                max_iter=self.max_iter,
                h_tol=self.h_tol,
                tabu_edges=tabu_edges_with_target,
            )

            if self.w_threshold:
                self.threshold_structural_model = linear.from_pandas(
                    X=self.X,
                    max_iter=self.max_iter,
                    h_tol=self.h_tol,
                    w_threshold=self.w_threshold,
                    tabu_edges=tabu_edges_with_target,
                )

        else:
            logger.info(
                f"Structural learning with {self.max_iter} max iterations and tolerance of {self.h_tol} "
                f"and lasso multiplier = {self.lasso_multiplier}"
            )
            self.structural_model = linear.from_pandas_lasso(
                X=self.X,
                max_iter=self.max_iter,
                beta=self.lasso_multiplier,
                h_tol=self.h_tol,
                tabu_edges=tabu_edges_with_target,
            )
            if self.w_threshold:
                self.threshold_structural_model = linear.from_pandas_lasso(
                    X=self.X,
                    max_iter=self.max_iter,
                    beta=self.lasso_multiplier,
                    h_tol=self.h_tol,
                    w_threshold=self.w_threshold,
                    tabu_edges=tabu_edges_with_target,
                )

    def _learn_non_linear_mlp_structure(self):
        """
        Learns the structure of DAG via NOTEARS algorithm by assuming MLP for any distribution type.
        """
        logger.info("Initiating Causal Discovery with MLP and combination of GLMs")
        logger.info("Defining restricted edges")

        tabu_edges_with_target = self._get_all_tabu_edges()

        logger.info(
            f"Structural learning with {self.max_iter} max iterations "
            f", lasso multiplier = {self.non_linear_args['lasso_multiplier']} "
            f"and ridge multiplier = {self.non_linear_args['ridge_multiplier']}"
        )

        self.structural_model, self.nonlinear_model = nonlinear.from_pandas(
            X=self.X,
            dist_type_schema=self.non_linear_args["feature_to_distribution_map"],
            lasso_beta=self.non_linear_args["lasso_multiplier"],
            ridge_beta=self.non_linear_args["ridge_multiplier"],
            use_bias=self.non_linear_args["use_bias"],
            hidden_layer_units=self.non_linear_args["hidden_layer_units"],
            max_iter=self.max_iter,
            h_tol=self.h_tol,
            tabu_edges=tabu_edges_with_target,
            use_gpu=self.non_linear_args["use_gpu"],
        )

        if self.w_threshold:
            logger.info(
                f"Structural learning with {self.max_iter} max iterations "
                f", lasso multiplier = {self.lasso_multiplier}, "
                f"ridge multiplier = {self.non_linear_args['ridge_multiplier']}"
                f" and hard thresholding at {self.w_threshold}"
            )
            self.threshold_structural_model, _ = nonlinear.from_pandas(
                X=self.X,
                dist_type_schema=self.non_linear_args["feature_to_distribution_map"],
                lasso_beta=self.non_linear_args["lasso_multiplier"],
                ridge_beta=self.non_linear_args["ridge_multiplier"],
                use_bias=self.non_linear_args["use_bias"],
                hidden_layer_units=self.non_linear_args["hidden_layer_units"],
                w_threshold=self.w_threshold,
                max_iter=self.max_iter,
                tabu_edges=tabu_edges_with_target,
                use_gpu=self.non_linear_args["use_gpu"],
                verbose=self.non_linear_args["verbose"],
            )

    def learn_structure(self):
        """
        Learns the structure of DAG via NOTEARS or Nonlinear NOTEARS-MLP algorithm.
        """
        if self.connections_type == "linear":
            self._learn_linear_structure()
        else:
            self._learn_non_linear_mlp_structure()
        logger.info("Structure learnt")

    def _get_graph_specs(self, structural_model, node_attributes):
        # Customising edges
        edge_attributes = {
            (u, v): {
                "width": np.abs(w)
                if np.abs(w) < 10
                else min(10, np.abs(w)),  # Scale edge width according to their weight
            }
            for u, v, w in structural_model.edges(data="weight")
        }

        target_edge_attributes = {
            (u, v): {
                "width": np.abs(w),  # Scale edges going to target width according to their weight
                "color": self._decide_edge_color_between_target(w),  # inflates churn thus bad(red)
                "shadow": {
                    "enabled": True,  # This adds a shadow underneath the edge
                    "color": "#FFD700",  # Choosing the color of the shadow
                },
            }
            for u, v, w in structural_model.edges(data="weight")
            if v == self.event_col
        }

        edge_attributes.update(target_edge_attributes)

        d = dict()
        for u, v, w in structural_model.edges(data="weight"):
            if v == self.event_col:
                d.update({(u, v): np.abs(w)})

        node_attributes.update(
            {
                max(d, key=d.get)[0]: {
                    "color": {
                        "background": self.higher_contribution_feature_color,
                        "highlight": {"background": "#ffcccc", "border": "#cce0ff"},
                    },
                    "shape": "hexagon",
                    "borderWidth": 2,
                    "size": 30,
                    "font": {
                        "color": "#FFFFFFD9",
                        "face": "Helvetica",
                    },
                },
                "fixed": {"x": True},
            }
        )

        return edge_attributes, node_attributes

    def _decide_edge_color_between_target(self, w):
        """
        Returns the colouring of an edge according
        to the coefficients sign. If we need
        to revert signs it will happen due to
        the invert_signs argument of the class.

        Parameters
        ----------
        w: float
            edge weight

        Returns
        -------
            str
            red or green
        """
        if self.invert_signs:
            return "green" if w < 0 else "red"
        else:
            return "green" if w > 0 else "red"

    def visualise_graph(self, path: Path):
        """
        Visualise resulting DAGS:
        1) Fully connected one
        2) Largest subgraph
        3) DAG under threshold cut off
        4) Largest subgraph after threshold cut off

        Parameters
        ----------
        path:  str
            folder path to save visualisations

        """

        opt = {
            "physics": {
                "solver": "repulsion",
                "repulsion": {
                    "nodeDistance": 400,
                    "springLength": 100,
                    "springConstant": 0.08,
                },
            }
        }

        all_edge_attributes = {
            "color": {
                "color": "#FFFFFFD9",
                "highlight": "#4a90e2d9",
            },
            "length": 10,
            "width": 1,
        }

        all_node_attributes = {
            "font": {
                "color": "#FFFFFFD9",
                "face": "Helvetica",
                "size": 20,
            },
            "shape": "box",
            "size": 15,
            "borderWidth": 2,
            "color": {"border": "#4a90e2d9", "background": "#001521"},
            "mass": 3,
        }

        node_attributes = {
            self.event_col: {
                "color": {
                    "background": self.event_color,
                    "highlight": {"background": "#ffcccc", "border": "#cce0ff"},
                },
                "shape": "circle",
                "size": 50,
                "label": self.event_label if self.event_label else self.event_col,
                "font": {
                    "color": "black",
                    "face": "Helvetica",
                },
                "fixed": {"y": True},
            }
        }

        edges, nodes = self._get_graph_specs(self.structural_model, node_attributes)

        fully_connected_viz = plot_structure(
            self.structural_model,
            all_node_attributes=all_node_attributes,
            all_edge_attributes=all_edge_attributes,
            node_attributes=nodes,
            edge_attributes=edges,
        )

        structural_model = self._require_structural_model()
        subgraph_viz = plot_structure(
            structural_model.get_largest_subgraph(),
            all_node_attributes=all_node_attributes,
            all_edge_attributes=all_edge_attributes,
            node_attributes=nodes,
            edge_attributes=edges,
        )

        logger.info("Saving fully connected graph")
        fully_connected_viz.show(str(path / "01_fully_connected.html"))
        subgraph_viz.set_options(options=json.dumps(opt))
        subgraph_viz.show(str(path / "02_largest_subgraph.html"))

        if self.w_threshold:
            logger.info(f"Pruning graph edges based on threshold w={self.w_threshold}")

            threshold_structural_model = self._require_threshold_structural_model()
            edges, nodes = self._get_graph_specs(threshold_structural_model, node_attributes)
            threshold_full_viz = plot_structure(
                threshold_structural_model,
                all_node_attributes=all_node_attributes,
                all_edge_attributes=all_edge_attributes,
                node_attributes=nodes,
                edge_attributes=edges,
            )
            logger.info("Threshold graph until it's a DAG")
            threshold_structural_model.threshold_till_dag()

            dag_full_viz = plot_structure(
                threshold_structural_model,
                all_node_attributes=all_node_attributes,
                all_edge_attributes=all_edge_attributes,
                node_attributes=nodes,
                edge_attributes=edges,
            )

            threshold_subgraph_full_viz = plot_structure(
                threshold_structural_model.get_largest_subgraph(),
                all_node_attributes=all_node_attributes,
                all_edge_attributes=all_edge_attributes,
                node_attributes=nodes,
                edge_attributes=edges,
            )

            if path:
                logger.info("Saving connected graph after pruning")
                threshold_full_viz.show(str(path / "03_threshold_graph.html"))
                logger.info("Saving DAG after pruning")
                dag_full_viz.set_options(options=json.dumps(opt))
                dag_full_viz.show(str(path / "04_threshold_DAG.html"))
                threshold_subgraph_full_viz.set_options(options=json.dumps(opt))
                threshold_subgraph_full_viz.show(str(path / "05_DAG_subgraph.html"))

    def get_edge_weights(self, path: Path):
        """
        Saves edge weights with directional
        arrows to a yaml file.

        Parameters
        ----------
        path: str
            folder path
        """
        structural_model = self._require_structural_model()
        edge_data_dict = dict()
        logger.info("Saving estimated edge weights")
        for u, v in structural_model.edges:
            weight = structural_model.get_edge_data(v=v, u=u)["weight"]
            logger.info(f"Estimated weight for {u}->{v}:{weight}")
            edge_data_dict.update({f"{u}->{v}": weight})
            with (path / "score_edges_full_structure.yml").open("w") as outfile:
                yaml.dump(edge_data_dict, outfile, default_flow_style=False)

        if self.w_threshold:
            threshold_structural_model = self._require_threshold_structural_model()
            edge_data_dict_thresh = dict()
            for u, v in threshold_structural_model.edges:
                weight = threshold_structural_model.get_edge_data(v=v, u=u)["weight"]
                logger.info(f"Threshold graph estimated weight for {u}->{v}:{weight}")
                edge_data_dict_thresh.update({f"{u}->{v}": weight})
            with (path / "score_edges_sparser_structure.yml").open("w") as outfile:
                yaml.dump(edge_data_dict_thresh, outfile, default_flow_style=False)

    def discover_dag(self, path: Path):
        """
        Run Structure learning for causal discovery process.

        Parameters
        ----------
        path:  Path
            folder path to save files
        """
        self.learn_structure()
        logger.info("Fetching edge weights")
        self.get_edge_weights(path=path)
        logger.info("Visualising causal graphs")
        self.visualise_graph(path=path)
        if self.connections_type != "linear":
            nonlinear_model = self._require_nonlinear_model()
            with (path / "non_linearMLP.pkl").open("wb") as file_pi:
                pickle.dump(self, file_pi)
            logger.info("Saving heatmaps of weights for the non linear model")
            all_weights = pd.DataFrame(nonlinear_model.adj_mean_effect)
            weight_names: list[str] = []
            if len(all_weights.columns) == len(self.X.columns):
                weight_names = list(self.X.columns)
            else:
                new_cols: list[str] = []
                for feat in self.X.columns:
                    if self.non_linear_args["feature_to_distribution_map"][feat] == "cat":
                        n_cats = self.X[feat].nunique()
                        weight_names.append(feat)
                        for idx in range(1, n_cats):
                            new_cols.append(f"{feat}{idx}")
                    else:
                        weight_names.append(feat)
                weight_names = weight_names + new_cols
            all_weights.columns = weight_names
            all_weights.index = weight_names

            all_std_errors = pd.DataFrame(nonlinear_model.adj_std_errors)
            all_std_errors.columns = weight_names
            all_std_errors.index = weight_names

            # Scale the weights to the range [0, 1]
            scaled_weights = (all_weights - all_weights.min()) / (
                all_weights.max() - all_weights.min()
            )

            # Extract the target_column
            last_column = all_weights[self.event_col]

            # Create a heatmap for the entire dataframe
            plt.figure(figsize=(22, 6))
            plt.subplot(1, 2, 1)
            sns.heatmap(
                scaled_weights.drop(self.event_col).drop(self.event_col, axis=1),
                cmap="coolwarm",
                annot=False,
                vmin=-1,
                vmax=1,
                center=0,
            )
            plt.title("All weights except target scaled")

            # Create a heatmap for the scaled last column
            plt.subplot(1, 2, 2)
            sns.heatmap(
                pd.DataFrame(last_column),
                cmap="coolwarm",
                annot=False,
                vmin=last_column.min(),
                vmax=last_column.max(),
                center=0,
            )
            plt.title("Target only non-scaled")

            plt.show()

            if path:
                all_weights.to_csv(path / "all_weights_mlp.csv")
                all_std_errors.to_csv(path / "all_std_error_weights_mlp.csv")
                plt.savefig(path / "all_weights_heatmap.png", bbox_inches="tight")
                plt.clf()

            if self.w_threshold:
                masked_weights = all_weights[np.abs(all_weights) >= self.w_threshold]
                scaled_weights_thresh = (masked_weights - masked_weights.min()) / (
                    masked_weights.max() - masked_weights.min()
                )
                # Extract the last column
                last_column = masked_weights[self.event_col]
                # Create a heatmap for the entire dataframe
                plt.figure(figsize=(22, 6))
                plt.subplot(1, 2, 1)
                sns.heatmap(
                    scaled_weights_thresh.drop(self.event_col).drop(self.event_col, axis=1),
                    cmap="coolwarm",
                    annot=False,
                    vmin=-1,
                    vmax=1,
                    center=0,
                )
                plt.title("All weights except target scaled with hard threshold")

                # Create a heatmap for the scaled last column
                plt.subplot(1, 2, 2)
                sns.heatmap(
                    pd.DataFrame(last_column),
                    cmap="coolwarm",
                    annot=False,
                    vmin=last_column.min(),
                    vmax=last_column.max(),
                    center=0,
                )
                plt.title("Target only non-scaled with hard threshold")

                plt.show()
                if path:
                    all_weights[np.abs(all_weights) >= self.w_threshold].to_csv(
                        path / "threshold_weights_mlp.csv"
                    )
                    all_std_errors[np.abs(all_weights) >= self.w_threshold].to_csv(
                        path / "threshold_std_error_weights_mlp.csv"
                    )

                    plt.savefig(
                        path / "all_weights_w_threshold_heatmap.png",
                        bbox_inches="tight",
                    )
                    plt.clf()
