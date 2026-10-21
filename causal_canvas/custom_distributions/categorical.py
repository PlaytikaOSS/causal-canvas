"""
Categorical distribution type for non-linear NOTEARS.

Defines :class:`DistTypeCategorical`, an implementation of the
``causalnex`` ``DistTypeBase`` interface that produces softmax-style
outputs and a one-hot loss for categorical-coded features used by the
MLP NOTEARS solver.

**Layer**: low-level. Plugged into the solver via
``dist_type_aliases["cat"]`` in the package ``__init__``.

**Relationship to causalnex**: subclasses
``causal_canvas.vendored.dist_type._base.DistTypeBase`` and uses
the upstream ``ExpandColumnsMixin`` for one-hot column expansion;
designed to compose with the rest of the upstream ``DistType*``
hierarchy.

See also
--------
:mod:`causal_canvas.custom_notears_mlp` — consumer of this dist type.
:mod:`causal_canvas.custom_distributions` — the dist-type registry.
"""

from copy import deepcopy
from typing import Optional

import numpy as np
import torch
from sklearn.preprocessing import OneHotEncoder
from torch import nn

from causal_canvas.vendored.dist_type._base import DistTypeBase, ExpandColumnsMixin
from causal_canvas.vendored.structure_model import StructureModel


class DistTypeCategorical(ExpandColumnsMixin, DistTypeBase):
    """Class defining categorical distribution type functionality"""

    # index group of categorical columns; populated during preprocess_X(fit_transform=True)
    idx_group: Optional[list[int]] = None
    # column expander for later preprocessing; populated during preprocess_X(fit_transform=True)
    encoder: Optional[OneHotEncoder] = None

    def _require_idx_group(self) -> list[int]:
        """Return a non-None idx_group or raise.

        Most methods on this class only make sense after preprocess_X has been
        called with ``fit_transform=True`` (which populates ``self.idx_group``).
        Using this helper at the top of each read-site gives a clear runtime
        error instead of a cryptic ``TypeError`` when idx_group is still None.
        """
        if self.idx_group is None:
            raise RuntimeError(
                "idx_group not initialized; first call preprocess_X with "
                "fit_transform=True before using this method"
            )
        return self.idx_group

    def get_columns(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Gets the column(s) associated with the instantiated DistType.

        Args:
            X: Full dataset to be selected from.

        Returns:
            1d or 2d np.ndarray of columns.
        """
        return X[:, self._require_idx_group()]

    def preprocess_X(self, X: np.ndarray, fit_transform: bool = True) -> np.ndarray:
        """
        Expands the feature dimension for each categorical column by:
        - One hot encode each of the categorical features
        - For each feature, get handle on groups of one-hot expanded columns
        - Store the handle groups
        - Return expanded array
        NOTE: the number of expanded columns is EQUAL to the number of classes
        for ease of use with the Pytorch loss functions.
        This is technically wasteful computationally (only need C-1 columns).

        Args:
            X: The original passed-in data.

            fit_transform: Whether the class first fits
            then transforms the data, or just transforms.
            Just transforming is used to preprocess new data after the
            initial NOTEARS fit.

        Returns:
            Preprocessed X
        """
        # deepcopy to prevent overwrite errors
        X = deepcopy(X)

        # fit the OneHotEncoder
        if fit_transform:
            self.encoder = OneHotEncoder(sparse_output=False, categories="auto", drop=None)
            self.encoder.fit(X[:, [self.idx]])

        # expand columns for this feature
        if self.encoder is None:
            raise RuntimeError(
                "encoder not initialized; first call preprocess_X with "
                "fit_transform=True before calling with fit_transform=False"
            )
        expanded_columns = self.encoder.transform(X[:, [self.idx]])
        # update the original column with the first expanded column
        X[:, self.idx] = expanded_columns[:, 0]
        # append the remainder cols to X
        X = self._expand_columns(X, expanded_columns[:, 1:])

        # update the idx_group with expanded columns
        if fit_transform:
            self.idx_group = []
            # preserve the first column location
            self.idx_group.append(self.idx)
            # the new cols are appended to the end of X contiguously
            n_new_cols = expanded_columns.shape[1] - 1
            idx_start = X.shape[1] - n_new_cols
            # preserve location of expanded columns
            self.idx_group += list(range(idx_start, X.shape[1]))

        return X

    def preprocess_tabu_edges(self, tabu_edges: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """
        Update tabu_edges taking into account expanded columns.

        Args:
            tabu_edges: The original tabu_edges.

        Returns:
            Preprocessed tabu_edges.
        """
        return self.update_tabu_edges(
            idx_group=self._require_idx_group(),
            tabu_edges=tabu_edges,
            tabu_idx_group=True,
        )

    def preprocess_tabu_nodes(self, tabu_nodes: list[int]) -> list[int]:
        """
        Update tabu_nodes taking into account expanded columns.

        Args:
            tabu_nodes: The original tabu_nodes.

        Returns:
            Preprocessed tabu_nodes.
        """
        return self.update_tabu_nodes(idx_group=self._require_idx_group(), tabu_nodes=tabu_nodes)

    def modify_h(self, square_weight_mat: torch.Tensor) -> torch.Tensor:
        """
        Used to prevent spurious cycles between expanded columns and other features.
        For example, A_cat1 -> B -> A_cat2 would not be penalized by the h(W) constraint.

        This modification solves that by adding the expanded columns of the
        squared adjacency matrix onto the original column. This effectively superimposes
        All expanded column connections onto a single connection

        Args:
            square_weight_mat: The squared adjacency matrix used in h(W).

        Returns:
            The modified W matrix.
        """
        idx_group = self._require_idx_group()
        orig_idx = idx_group[0]
        expand_idx = idx_group[1:]

        # Add on the edges from expanded nodes.
        square_weight_mat[orig_idx, :] = square_weight_mat[orig_idx, :] + torch.sum(
            square_weight_mat[expand_idx, :], dim=0
        )
        # Add on the edges to expanded nodes.
        square_weight_mat[:, orig_idx] = square_weight_mat[:, orig_idx] + torch.sum(
            square_weight_mat[:, expand_idx], dim=1
        )

        return square_weight_mat

    def collapse_adj(self, adj: np.ndarray) -> np.ndarray:
        """
        Collapse categories into one column through summation.
        Conceptually the same as modify_h.

        Args:
            adj: The adjacency matrix.

        Returns:
            Updated adjacency matrix.
        """
        idx_group = self._require_idx_group()
        orig_idx = idx_group[0]
        expand_idx = idx_group[1:]

        # Add on the edges from expanded nodes.
        adj[orig_idx, :] = adj[orig_idx, :] + np.sum(adj[expand_idx, :], axis=0)
        # Add on the edges to expanded nodes.
        adj[:, orig_idx] = adj[:, orig_idx] + np.sum(adj[:, expand_idx], axis=1)

        return adj

    @staticmethod
    def _to_index(X_one_hot: torch.Tensor) -> torch.Tensor:
        """
        Recover the numerical columns by argmaxing a one-hot vector.

        Args:
            X_one_hot: The one-hot tensor to be collapsed.

        Returns:
            A 1d tensor representing the classes defined by the above one-hot
            tensor.
        """
        return torch.argmax(X_one_hot, dim=1)

    def add_to_node(self, sm: StructureModel) -> StructureModel:
        """
        Adds self to a node of a structure model corresponding to
        all indexes in self.idx_group.

        Args:
            sm: The input StructureModel

        Returns:
            Updated StructureModel
        """
        for idx in self._require_idx_group():
            sm.nodes[idx]["dist_type"] = self
        return sm

    def loss(self, X: torch.Tensor, X_hat: torch.Tensor) -> torch.Tensor:
        """
        Uses the functional implementation of the CrossEntropyLoss class
        https://pytorch.org/docs/master/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss.

        Returns the mean row wise cross entropy loss for a single group of categorical columns.

        NOTE: the pytorch implementation assumes a numeric target input.
        Therefore, collapse the one hot columns into a numeric target column.

        Args:
            X: The original data passed into NOTEARS (i.e. the reconstruction target).

            X_hat: The reconstructed data.

        Returns:
            Scalar pytorch tensor of the reconstruction loss between X and X_hat.
        """

        idx_group = self._require_idx_group()
        return nn.functional.cross_entropy(
            input=X_hat[:, idx_group],
            target=self._to_index(X[:, idx_group]),
            reduction="mean",
        )

    def inverse_link_function(self, X_hat: torch.Tensor) -> torch.Tensor:
        """
        Softmax inverse link function for categorical data.

        Args:
            X_hat: Reconstructed data in the latent space.

        Returns:
            Modified X_hat.
            MUST be same shape as passed in data.
            Projects the self.idx column from the latent space to the dist_type space.
        """
        idx_group = self._require_idx_group()
        X_hat[:, idx_group] = torch.softmax(X_hat[:, idx_group], dim=1)
        return X_hat

    @staticmethod
    def make_node_name(colname: str, catidx: int) -> str:
        """
        Renaming scheme for expanded categorical columns.
        NOTE: column is not renamed if catidx is 0.
        This is bc original column name needs to stay constant.

        Args:
            colname: The base column used in the renaming.

            catidx: The index of the categorical expansion.

        Returns:
            Updated column name.
        """
        if catidx:
            return f"{colname}{catidx}"
        return colname

    def update_idx_col(self, idx_col: dict[int, str]) -> dict[int, str]:
        """
        Expand the named columns to include category names.

        Args:
            idx_col: The original index to column mapping.

        Returns:
            Updated index to column mapping.
        """
        idx_group = self._require_idx_group()
        new_idx_cols = {}
        colname = idx_col.pop(idx_group[0])
        for catidx, idx in enumerate(idx_group):
            new_idx_cols[idx] = self.make_node_name(colname, catidx)
        return {**idx_col, **new_idx_cols}
