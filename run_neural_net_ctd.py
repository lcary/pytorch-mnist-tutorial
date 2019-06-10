
"""
A simple convolutional neural network that uses PyTorch,
trained to recognize handwritten digits using the MNIST dataset.

Tutorial:
https://nextjournal.com/gkoehler/pytorch-mnist
"""
import json
import os

import torch
from torch import optim

from lib.args import get_common_parser
from lib.network import train, test, TrainingState, Net, mnist_data_loader
from lib.utils import save_loss_data_file


def main() -> None:
    parser = get_common_parser()
    parser.add_argument('previous_loss_data', help='previous loss data file')
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.previous_loss_data, 'r') as infile:
        prev_loss_data = json.load(infile)

    continued_network = Net()
    continued_optimizer = optim.SGD(
        continued_network.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum)

    network_state_dict = torch.load(os.path.join(args.out_dir, "model.pth"))
    continued_network.load_state_dict(network_state_dict)

    optimizer_state_dict = torch.load(os.path.join(args.out_dir, "optimizer.pth"))
    continued_optimizer.load_state_dict(optimizer_state_dict)

    previous_epochs = prev_loss_data['epochs']
    start_range = previous_epochs + 1
    end_range = start_range + args.n_epochs

    random_seed = 1  # this should be a random func in non-demo code
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)

    # Create loaders for the training and test data

    train_loader = mnist_data_loader(args.data_dir, args.batch_size_train)
    test_loader = mnist_data_loader(args.data_dir, args.batch_size_test, is_training=False)

    train_losses = prev_loss_data['train_losses']
    train_counter = prev_loss_data['train_counter']
    test_losses = prev_loss_data['test_losses']
    test_counter = prev_loss_data['test_counter']
    test_counter.extend([i * len(train_loader.dataset) for i in range(start_range, end_range)])

    training_state = TrainingState(train_losses, train_counter, args.out_dir)
    test(test_loader, continued_network, test_losses)
    for epoch in range(start_range, end_range):
        train(train_loader, continued_network, continued_optimizer, epoch, args.log_interval, training_state)
        test(test_loader, continued_network, test_losses)

    total_num_epochs = previous_epochs + args.n_epochs
    save_loss_data_file(args.out_dir, train_counter, train_losses, test_counter, test_losses, total_num_epochs)


if __name__ == '__main__':
    main()
