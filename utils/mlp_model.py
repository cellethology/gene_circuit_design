"""
MLP (Multi-Layer Perceptron) model using PyTorch and skorch for regression tasks
"""

import torch
import torch.nn as nn
from skorch import NeuralNetRegressor


class MLPRegressor(nn.Module):
    """
    Multi-Layer Perceptron for regression tasks.

    Parameters
    ----------
    input_size : int
        Number of input features
    hidden_sizes : list of int, default=[256, 128, 64]
        List of hidden layer sizes
    dropout : float, default=0.2
        Dropout probability
    """

    def __init__(self, input_size, hidden_sizes=None, dropout=0.2):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [256, 128, 64]

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.dropout_prob = dropout

        # Build the network layers
        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.extend(
                [nn.Linear(prev_size, hidden_size), nn.ReLU(), nn.Dropout(dropout)]
            )
            prev_size = hidden_size

        # Output layer (single output for regression)
        layers.append(nn.Linear(prev_size, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_size)

        Returns
        -------
        torch.Tensor
            Output predictions of shape (batch_size, 1)
        """
        return self.network(x).squeeze(-1)


def create_mlp_regressor(
    input_size,
    hidden_sizes=None,
    dropout=0.2,
    max_epochs=100,
    lr=0.01,
    batch_size=32,
    random_state=42,
    device="cpu",
):
    """
    Create a scikit-learn compatible MLP regressor using skorch.

    Parameters
    ----------
    input_size : int
        Number of input features
    hidden_sizes : list of int, default=None
        List of hidden layer sizes. If None, uses [256, 128, 64]
    dropout : float, default=0.2
        Dropout probability
    max_epochs : int, default=100
        Maximum number of training epochs
    lr : float, default=0.01
        Learning rate for optimizer
    batch_size : int, default=32
        Batch size for training
    random_state : int, default=42
        Random seed for reproducibility
    device : str, default='cpu'
        Device to run the model on ('cpu' or 'cuda')

    Returns
    -------
    NeuralNetRegressor
        Scikit-learn compatible neural network regressor
    """
    if hidden_sizes is None:
        hidden_sizes = [256, 128, 64]

    # Set random seeds for reproducibility
    torch.manual_seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_state)

    net = NeuralNetRegressor(
        module=MLPRegressor,
        module__input_size=input_size,
        module__hidden_sizes=hidden_sizes,
        module__dropout=dropout,
        max_epochs=max_epochs,
        lr=lr,
        batch_size=batch_size,
        optimizer=torch.optim.Adam,
        criterion=nn.MSELoss,
        device=device,
        verbose=0,  # Set to 1 to see training progress
        random_state=random_state,
    )

    return net


def create_adaptive_mlp_regressor(input_size, random_state=42, device="cpu"):
    """
    Create an adaptive MLP regressor that adjusts architecture based on input size.

    Parameters
    ----------
    input_size : int
        Number of input features
    random_state : int, default=42
        Random seed for reproducibility
    device : str, default='cpu'
        Device to run the model on

    Returns
    -------
    NeuralNetRegressor
        Adaptive MLP regressor optimized for the input size
    """
    # Adaptive architecture based on input size
    if input_size < 50:
        hidden_sizes = [64, 32]
        max_epochs = 150
        lr = 0.01
    elif input_size < 500:
        hidden_sizes = [128, 64, 32]
        max_epochs = 100
        lr = 0.005
    elif input_size < 2000:
        hidden_sizes = [256, 128, 64]
        max_epochs = 100
        lr = 0.001
    else:
        hidden_sizes = [512, 256, 128, 64]
        max_epochs = 80
        lr = 0.001

    return create_mlp_regressor(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        dropout=0.3,
        max_epochs=max_epochs,
        lr=lr,
        batch_size=64,
        random_state=random_state,
        device=device,
    )
