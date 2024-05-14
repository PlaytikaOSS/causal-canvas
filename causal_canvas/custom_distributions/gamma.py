import numpy as np
import torch
from causalnex.structure.pytorch.dist_type._base import DistTypeBase


class DistTypeGamma(DistTypeBase):
    """Class defining gamma distribution type functionality.

    Methods
    -------
    preprocess_X(X, fit_transform=True)
        Perform positivity check on the original data.

    loss(X, X_hat)
        Compute the negative log-likelihood loss for gamma GLM.

    inverse_link_function(X_hat)
        Exponential inverse link function for gamma distribution.

    """

    def preprocess_X(self, X: np.ndarray, fit_transform: bool = True) -> np.ndarray:
        """Perform positivity check on the original data.

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

        Raises
        ------
        ValueError
            If data has non-positive values.
        """
        if (X[:, self.idx] <= 0).sum() > 0:
            raise ValueError("All data must be positive for the gamma distribution.")
        return X

    def loss(self, X: torch.Tensor, X_hat: torch.Tensor) -> torch.Tensor:
        """Compute the negative log-likelihood loss for gamma GLM. This parameterisation
        is an alternative one for which we use the mean and shape, frequently used for in GLM.
        GLM estimate the mean, while the mean under Gamma distribution is defined as
        \mu=E(X)=\alpha/\beta. By replacing the rate, we get Gamma(\mu, \alpha). In GLM
        we assume he shape to be a constant, thus a \phi dispersion parameter needs to
        be assumed: \phi=1/\alpha. Thus, we need to maximise the Gamma log-likelihood
        based on these parameters.

        Parameters
        ----------
        X : torch.Tensor
            The original data passed into NOTEARS (i.e., the reconstruction target).
        X_hat : torch.Tensor
            The reconstructed data.

        Returns
        -------
        torch.Tensor
            Scalar PyTorch tensor of the reconstruction loss between X and X_hat.
        """
        log_mu = X_hat[:, self.idx]
        mu = torch.exp(log_mu)  # Recover the mean from the log
        phi = 1  # Dispersion parameter, can be learned as well

        # Compute log-likelihood for the gamma distribution
        log_likelihood = (
            torch.lgamma(X[:, self.idx] / phi)
            + (1 / phi - 1) * torch.log(X[:, self.idx])
            - (X[:, self.idx] / mu)
        )

        return -log_likelihood.mean()

    def inverse_link_function(self, X_hat: torch.Tensor) -> torch.Tensor:
        """
        Exponential inverse link function for gamma distribution.
        The link function for gamma distribution is the logarithm,
        thus we need to reconstruct back.

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
        X_hat[:, self.idx] = torch.exp(X_hat[:, self.idx])
        return X_hat
