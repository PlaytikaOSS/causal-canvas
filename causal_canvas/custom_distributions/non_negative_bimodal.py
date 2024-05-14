import torch
import torch.nn as nn
from causalnex.structure.pytorch.dist_type._base import DistTypeBase


class DistTypeNonNegBimodal(DistTypeBase):
    """Class defining bimodal non-negative defined Gaussian distribution type functionality.

    Parameters
    ----------
    idx : int
        Index representing the position of the feature in the data.
    init_parameters : bool, optional, default: True
        Flag indicating whether to initialize parameters randomly.

    Attributes
    ----------
    alpha : nn.Parameter
        Learnable parameter representing the mixture ratio between two modes.
    mean1 : nn.Parameter
        Learnable parameter representing the mean of the first mode.
    mean2 : nn.Parameter
        Learnable parameter representing the mean of the second mode.
    std : nn.Parameter
        Learnable parameter representing the standard deviation of the Gaussian distribution.

    Methods
    -------
    preprocess_X(X, fit_transform=True)
        Preprocesses the input data.

    log_likelihood(X, X_hat, mean, std)
        Computes the log-likelihood for a Gaussian distribution.

    loss(X, X_hat)
        Computes the average non-negative Gaussian loss.

    inverse_link_function(X_hat)
        Applies the inverse link function to the reconstructed data.

    softplus(x)
        Softplus function.

    """

    def __init__(self, idx, init_parameters=True):
        super().__init__(idx)
        self.alpha = nn.Parameter(torch.clamp(torch.rand(1), min=0.0, max=1.0))
        self.mean1 = nn.Parameter(torch.rand(1))
        self.mean2 = nn.Parameter(torch.rand(1))
        self.std = nn.Parameter(torch.clamp(torch.rand(1), min=1e-6))

        if not init_parameters:
            # If not initializing parameters randomly, fix them for evaluation
            for param in [self.alpha, self.mean1, self.mean2]:
                param.requires_grad = False

    def preprocess_X(self, X, fit_transform=True):
        """Preprocesses the input data.

        Parameters
        ----------
        X : torch.Tensor
            Input data.
        fit_transform : bool, optional, default: True
            Flag indicating whether to fit and transform the data.

        Returns
        -------
        torch.Tensor
            Preprocessed data.
        """
        # Your preprocessing logic here
        return X

    def log_likelihood(self, X, X_hat, mean, std):
        """Computes the log-likelihood for a Gaussian distribution.

        Parameters
        ----------
        X : torch.Tensor
            Original data.
        X_hat : torch.Tensor
            Reconstructed data.
        mean : nn.Parameter
            Mean parameter.
        std : nn.Parameter
            Standard deviation parameter.

        Returns
        -------
        torch.Tensor
            Log-likelihood for the Gaussian distribution.
        """
        std = torch.tensor(std)
        log_likelihood = -0.5 * (
            ((X[:, self.idx] - mean) / std) ** 2 + 2 * torch.log(std)
        )
        return torch.tensor(log_likelihood.mean())

    def loss(self, X, X_hat):
        """Computes the average non-negative Gaussian loss.

        Parameters
        ----------
        X : torch.Tensor
            Original data.
        X_hat : torch.Tensor
            Reconstructed data.

        Returns
        -------
        torch.Tensor
            Scalar PyTorch tensor of the reconstruction loss between X and X_hat.
        """
        # Log likelihoods for the two modes
        log_likelihood1 = self.log_likelihood(X, X_hat, self.mean1, self.std)
        log_likelihood2 = self.log_likelihood(X, X_hat, self.mean2, self.std)

        # Combine the losses with learnable mixture ratio
        bimodal_loss = -(
            self.alpha * log_likelihood1 + (1 - self.alpha) * log_likelihood2
        )

        return bimodal_loss.mean()

    def inverse_link_function(self, X_hat):
        """Applies the inverse link function to the reconstructed data.

        Parameters
        ----------
        X_hat : torch.Tensor
            Reconstructed data.

        Returns
        -------
        torch.Tensor
            Reconstructed data after applying the inverse link function.
        """
        # Calculate the inverse link as a weighted sum of the means
        inverse_link = self.alpha * self.mean1 + (1 - self.alpha) * self.mean2
        X_hat[:, self.idx] = self.softplus(inverse_link)
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
