"""
Usage:

    python graph.py

or:

    python graph.py -i <loss_data_filepath>

Expects a file at out/loss_data.json with following format:

    data = {
        'train_counter': list<int>,
        'train_losses': list<int>,
        'test_counter': list<int>,
        'test_losses': list<int>,
    }

"""
import argparse
import json
import os

import matplotlib.pyplot as plt

os.makedirs('out', exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input-filepath')
args = parser.parse_args()

if args.input_filepath:
    loss_data_file = args.input_filepath
else:
    loss_data_file = os.path.join('out', 'loss_data.json')

with open(loss_data_file, 'r') as infile:
    data = json.load(infile)

# Plot the training curve:

fig = plt.figure()
plt.plot(data['train_counter'], data['train_losses'], color='blue')
plt.scatter(data['test_counter'], data['test_losses'], color='red')
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.xlabel('number of training examples seen')
plt.ylabel('negative log likelihood loss')
fpath = os.path.join('out', 'performance_graph.png')
print(f'saving graph of model performance to {fpath}')
fig.savefig(fpath, dpi=fig.dpi)
