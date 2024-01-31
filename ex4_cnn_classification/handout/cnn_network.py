import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class CNN(nn.Module):
    """Convolutional Neural Network.
    
    We provide a simple network with a Conv layer, followed by pooling,
    and a fully connected layer. Modify this to test different architectures,
    and hyperparameters, i.e. different number of layers, kernel size, feature
    dimensions etc.

    See https://pytorch.org/docs/stable/nn.html for a list of different layers
    in PyTorch.
    """

    def __init__(self):
        """Initialize layers."""
        super().__init__()

        self.activ = nn.ReLU
        size = 12#15
        self.in_pipes = nn.Sequential(
                nn.BatchNorm2d(3),
                nn.Conv2d(3, size, 8, padding_mode="reflect"),
                nn.MaxPool2d(5,5),
                nn.Conv2d(size, size, 8, padding_mode="reflect"),
                nn.GELU(),
                #nn.MaxPool2d(2, 2),
                nn.Flatten(1),
        )
        # experimentally found values bc I did not want to do the calculations.
        in_size = size
        m=480
        out=6
        self.out = nn.Sequential(
            nn.Linear(in_size, m, bias=True),
            nn.ReLU(),
            nn.Linear(m, out),
            )
        #self.out = nn.Identity()
    
    def forward(self, x):
        """Forward pass of network."""
        x = self.in_pipes(x)

        x = self.out(x)

        return x

    def write_weights(self, fname):
        """ Store learned weights of CNN """
        torch.save(self.state_dict(), fname)

    def load_weights(self, fname):
        """
        Load weights from file in fname.
        The evaluation server will look for a file called checkpoint.pt
        """
        ckpt = torch.load(fname)
        self.load_state_dict(ckpt)


def get_loss_function():
    """Return the loss function to use during training. We use
       the Cross-Entropy loss for now.
    
    See https://pytorch.org/docs/stable/nn.html#loss-functions.
    """
    return nn.CrossEntropyLoss()


def get_optimizer(network, lr=0.001, momentum=0.9):
    """Return the optimizer to use during training.
    
    network specifies the PyTorch model.

    See https://pytorch.org/docs/stable/optim.html#how-to-use-an-optimizer.
    """

    # The fist parameter here specifies the list of parameters that are
    # learnt. In our case, we want to learn all the parameters in the network
    return optim.Adadelta(network.parameters(), lr=lr, weight_decay=1e-2, rho=0.85)
