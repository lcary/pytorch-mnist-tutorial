
"""
A simple convolutional neural network that uses PyTorch,
trained to recognize handwritten digits using the MNIST dataset.

Tutorial:
https://nextjournal.com/gkoehler/pytorch-mnist
"""
import json
import os

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt

os.makedirs('data', exist_ok=True)
os.makedirs('out', exist_ok=True)

n_epochs = 3
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)


def mnist_data_loader(batch_size, is_training=True):
    return torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            'data/', train=is_training, download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,))
            ])
        ), batch_size=batch_size, shuffle=True)



# Create loaders for the training and test data

train_loader = mnist_data_loader(batch_size_train)
test_loader = mnist_data_loader(batch_size_test, is_training=False)

# Visualize some examples of the training data

examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)

fig = plt.figure()
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.tight_layout()
    plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
    plt.title("Ground Truth: {}".format(example_targets[i]))
    plt.xticks([])
    plt.yticks([])
fpath = os.path.join('out', 'example_labeled_data.png')
print(f'saving examples of training data to {fpath}')
fig.savefig(fpath, dpi=fig.dpi)


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


# Train our model and test every epoch

network = Net()
optimizer = optim.SGD(
    network.parameters(),
    lr=learning_rate,
    momentum=momentum)

train_losses = []
train_counter = []
test_losses = []
test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]


def train(network, optimizer, epoch):
    network.train()

    for batch_idx, (data, target) in enumerate(train_loader):

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
                len(train_loader.dataset),
                100.0 * batch_idx / len(train_loader),
                loss.item()))
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))
            torch.save(network.state_dict(), 'out/model.pth')
            torch.save(optimizer.state_dict(), 'out/optimizer.pth')


def test(network, optimizer):
    network.eval()
    test_loss = 0
    correct = 0

    # use no_grad to avoid storing computations made to produce the output
    # in the computation graph
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss,
        correct,
        len(test_loader.dataset),
        100.0 * correct / len(test_loader.dataset)))


test(network, optimizer)
for epoch in range(1, n_epochs + 1):
    train(network, optimizer, epoch)
    test(network, optimizer)

data = {
    'train_counter': train_counter,
    'train_losses': train_losses,
    'test_counter': test_counter,
    'test_losses': test_losses,
}

loss_data_file = os.path.join('out', 'loss_data.json')
with open(loss_data_file, 'w') as outfile:
    json.dump(data, outfile, indent=2)

# Check the model's output

with torch.no_grad():
    output = network(example_data)

fig = plt.figure()
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.tight_layout()
    plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
    plt.title("Prediction: {}".format(
        output.data.max(1, keepdim=True)[1][i].item()))
    plt.xticks([])
    plt.yticks([])
fpath = os.path.join('out', 'example_model_predictions.png')
print(f'saving example model predictions to {fpath}')
fig.savefig(fpath, dpi=fig.dpi)
