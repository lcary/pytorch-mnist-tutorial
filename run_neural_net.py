
"""
A simple convolutional neural network that uses PyTorch,
trained to recognize handwritten digits using the MNIST dataset.

Tutorial:
https://nextjournal.com/gkoehler/pytorch-mnist
"""
import os
from typing import Any

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch import nn as nn
from torch.utils.data import DataLoader

from lib.args import get_args
from lib.network import mnist_data_loader, Net, train, test, TrainingState
from lib.utils import save_loss_data_file


def save_example_training_data(loader: DataLoader) -> Any:
    """
    Visualize some examples of the training data
    """
    examples = enumerate(loader)
    batch_idx, (example_data, example_targets) = next(examples)
    fig = plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Ground Truth: {}".format(example_targets[i]))
        plt.xticks([])
        plt.yticks([])
    fpath = os.path.join('out', 'example_labeled_data.png')
    print(f'saving examples of training data to {fpath}')
    fig.savefig(fpath, dpi=fig.dpi)
    return example_data


def save_example_prediction_data(output_dir: str, network: nn.Module, example_data: Any) -> None:
    # Check the model's output

    with torch.no_grad():
        output = network(example_data)

    fig = plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Prediction: {}".format(
            output.data.max(1, keepdim=True)[1][i].item()))
        plt.xticks([])
        plt.yticks([])
    fpath = os.path.join(output_dir, 'example_model_predictions.png')
    print(f'saving example model predictions to {fpath}')
    fig.savefig(fpath, dpi=fig.dpi)


def main() -> None:
    args = get_args()

    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    n_epochs = args.n_epochs
    batch_size_train = args.batch_size_train
    batch_size_test = args.batch_size_test
    learning_rate = args.learning_rate
    momentum = args.momentum
    log_interval = args.log_interval

    random_seed = 1  # this should be a random func in non-demo code
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)

    # Create loaders for the training and test data

    train_loader = mnist_data_loader(args.data_dir, batch_size_train)
    test_loader = mnist_data_loader(args.data_dir, batch_size_test, is_training=False)

    example_data = save_example_training_data(train_loader)

    # Train our model and test every epoch

    network = Net()
    optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]

    training_state = TrainingState(train_losses, train_counter, args.out_dir)
    test(test_loader, network, test_losses)
    for epoch in range(1, n_epochs + 1):
        train(train_loader, network, optimizer, epoch, log_interval, training_state)
        test(test_loader, network, test_losses)

    save_loss_data_file(args.out_dir, train_counter, train_losses, test_counter, test_losses)

    save_example_prediction_data(args.out_dir, network, example_data)


if __name__ == '__main__':
    main()
