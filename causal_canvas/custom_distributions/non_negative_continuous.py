"""
Non-negative continuous distribution type for non-linear NOTEARS.

Defines :class:`DistTypeNonNegativeContinuous`, an implementation of
the ``causalnex`` ``DistTypeBase`` interface for strictly non-negative
continuous features (uses a softplus-style link to enforce the
non-negativity constraint).

**Layer**: low-level. Plugged into the solver via
``dist_type_aliases["pos_cont"]`` in the package ``__init__``.

**Relationship to causalnex**: subclasses
``causal_canvas.vendored.dist_type._base.DistTypeBase`` and slots
into the same registry as the upstream ``DistType*`` classes.

See also
--------
:mod:`causal_canvas.custom_notears_mlp` — consumer of this dist type.
:mod:`causal_canvas.custom_distributions` — the dist-type registry.
"""

import torch

from causal_canvas.vendored.dist_type._base import DistTypeBase


class DistTypeNonNegativeContinuous(DistTypeBase):
    """Class defining non-negative continuous distribution type functionality.

    Methods
    -------
    loss(X, X_hat)
        Computes the average non-negative Gaussian loss.

    inverse_link_function(X_hat)
        Applies the softplus inverse link function for non-negative continuous data.

    softplus(x)
        Softplus function.

    """

    def loss(self, X: torch.Tensor, X_hat: torch.Tensor) -> torch.Tensor:
        """Computes the average non-negative Gaussian loss.

        Parameters
        ----------
        X : torch.Tensor
            Original data passed into NOTEARS (i.e., the reconstruction target).
        X_hat : torch.Tensor
            Reconstructed data.

        Returns
        -------
        torch.Tensor
            Scalar PyTorch tensor of the reconstruction loss between X and X_hat.
        """
        return (0.5 / X.shape[0]) * torch.sum((torch.exp(X_hat[:, self.idx]) - X[:, self.idx]) ** 2)

    def inverse_link_function(self, X_hat: torch.Tensor) -> torch.Tensor:
        """Applies the softplus inverse link function for non-negative continuous data.

        Parameters
        ----------
        X_hat : torch.Tensor
            Reconstructed data in the latent space.

        Returns
        -------
        torch.Tensor
            Modified X_hat. Must be the same shape as the passed-in data.
            Projects the self.idx column from the latent space to the dist_type space.
        """
        X_hat[:, self.idx] = self.softplus(X_hat[:, self.idx])
        return X_hat

    def softplus(self, x: torch.Tensor) -> torch.Tensor:
        """Softplus function.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor after applying the softplus function.
        """
        return torch.log(1 + torch.exp(x))
