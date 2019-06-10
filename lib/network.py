import os
from typing import List

import torch
import torchvision
from torch import nn as nn
from torch.nn import functional as F
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader


def mnist_data_loader(
        data_dir: str,
        batch_size: int,
        is_training: bool = True) -> DataLoader:

    return DataLoader(
        torchvision.datasets.MNIST(
            data_dir, train=is_training, download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,))
            ])
        ), batch_size=batch_size, shuffle=True)


class Net(nn.Module):
    """
    A convolutional neural net designed to recognize characters
    from the MNIST data set.

    Layers:
      - two 2D convolutional layers
      - two fully-connected (linear) layers

    Activation function:
      - ReLUs

    Regularization:
      - 2 dropout layers
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


class TrainingState(object):
    """
    Class for keeping track of the neural network state during training.
    """

    def __init__(self, losses: List[float], counter: List[int], output_dir: str) -> None:
        self.losses = losses
        self.counter = counter
        self.model_pth = os.path.join(output_dir, 'model.pth')
        self.optimizer_pth = os.path.join(output_dir, 'optimizer.pth')

    def update(self, network: nn.Module, optimizer: Optimizer, loss: float, count: int) -> None:
        self.losses.append(loss)
        self.counter.append(count)
        torch.save(network.state_dict(), self.model_pth)
        torch.save(optimizer.state_dict(), self.optimizer_pth)


def train(
        loader: DataLoader,
        network: nn.Module,
        optimizer: Optimizer,
        epoch: int,
        log_interval: int,
        state: TrainingState) -> None:

    network.train()

    for batch_idx, (data, target) in enumerate(loader):

        # manually set all gradients to zero
        optimizer.zero_grad()

        # produce the network's output (forward pass)
        output = network(data)

        # compute negative log-likelihood loss between
        # the output and the ground truth label
        loss = F.nll_loss(output, target)

        # collect a new set of gradients and
        # backprogpagate to network parameters
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:0f}%)]\tLoss: {:.6f}'.format(
                epoch,
                batch_idx * len(data),
                len(loader.dataset),
                100.0 * batch_idx / len(loader),
                loss.item()))
            count = (batch_idx * 64) + ((epoch - 1) * len(loader.dataset))
            state.update(network, optimizer, loss.item(), count)


def test(loader: DataLoader, network: nn.Module, losses: List[float]) -> None:
    network.eval()
    test_loss = 0
    correct = 0

    # use no_grad to avoid storing computations made to produce the output
    # in the computation graph
    with torch.no_grad():
        for data, target in loader:
            output = network(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()

    test_loss /= len(loader.dataset)
    losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss,
        correct,
        len(loader.dataset),
        100.0 * correct / len(loader.dataset)))
