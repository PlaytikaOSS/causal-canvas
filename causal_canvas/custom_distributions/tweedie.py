import numpy as np
import torch
import torch.nn as nn
from causalnex.structure.pytorch.dist_type._base import DistTypeBase


class DistTypeTweedie(DistTypeBase):
    """Class defining Tweedie distribution type functionality.

    Methods
    -------
    preprocess_X(X, fit_transform=True)
        Perform non-negativity check on the original data.

    loss(X, X_hat, p=1.5)
        Compute the negative log-likelihood loss for the Tweedie GLM.

    gamma_loss(X, X_hat)
        Compute the negative log-likelihood loss for gamma GLM.

    inverse_link_function(X_hat)
        Exponential inverse link function for the Tweedie distribution.

    """

    def preprocess_X(self, X: np.ndarray, fit_transform: bool = True) -> np.ndarray:
        """Perform non-negativity check on the original data.

        Parameters
        ----------
        X : np.ndarray
            The original passed-in data.
        fit_transform : bool, optional
            Whether the class first fits then transforms the data, or just transforms.
            Just transforming is used to preprocess new data after the initial NOTEARS fit.

        Returns
        -------
        np.ndarray
            Preprocessed X.
        """
        if (X[:, self.idx] < 0).sum() > 0:
            raise ValueError(
                "All data must be non-negative for the Tweedie distribution."
            )
        return X

    def loss(
        self, X: torch.Tensor, X_hat: torch.Tensor, p: float = 1.5
    ) -> torch.Tensor:
        """Compute the negative log-likelihood loss for the Tweedie GLM.

        Parameters
        ----------
        X : torch.Tensor
            The original data passed into NOTEARS (i.e., the reconstruction target).
        X_hat : torch.Tensor
            The reconstructed data.
        p : float, optional
            Power parameter for the Tweedie distribution.

        Returns
        -------
        torch.Tensor
            Scalar PyTorch tensor of the reconstruction loss between X and X_hat.
        """
        log_mu = X_hat[:, self.idx]
        mu = torch.exp(log_mu)  # Recover the mean from the log

        # Poisson loss
        poisson_loss = nn.functional.poisson_nll_loss(
            input=log_mu,
            target=X[:, self.idx],
            reduction="mean",
            log_input=True,
            full=False,
        )

        self.p = p
        # Gamma loss
        gamma_loss = self.gamma_loss(X, X_hat)

        # Combine Poisson and Gamma losses based on the Tweedie power parameter
        tweedie_loss = (
            (1 / (p * (p - 1))) * ((mu**p) / p - mu * X[:, self.idx])
            + poisson_loss
            + gamma_loss
        )

        return tweedie_loss.mean()

    def gamma_loss(self, X: torch.Tensor, X_hat: torch.Tensor) -> torch.Tensor:
        """Compute the negative log-likelihood loss for gamma GLM.

        Parameters
        ----------
        X : torch.Tensor
            The original data passed into NOTEARS (i.e., the reconstruction target).
        X_hat : torch.Tensor
            The reconstructed data.

        Returns
        -------
        torch.Tensor
            Scalar PyTorch tensor of the gamma loss between X and X_hat.
        """
        log_mu = X_hat[:, self.idx]
        mu = torch.exp(log_mu)  # Recover the mean from the log
        phi = 1  # Dispersion parameter, can be learned as well

        # Compute log-likelihood for the gamma distribution
        log_likelihood = (
            # Account for the zeroes to be very small numbers for computational purposes
            torch.lgamma((X[:, self.idx] + 1e-6) / phi)
            + (1 / phi - 1) * torch.log(X[:, self.idx] + 1e-6)
            - ((X[:, self.idx] + 1e-6) / mu)
        )
        return -log_likelihood.mean()

    def inverse_link_function(self, X_hat: torch.Tensor) -> torch.Tensor:
        """Exponential inverse link function for Tweedie distribution.

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
        X_hat[:, self.idx] = (
            X_hat[:, self.idx].pow(2 - self.p) / (1 - self.p)
        ).exp() / (1 - self.p)
        return X_hat
