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

from lib.utils import get_timestamp_str

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('loss_data_file', help='previous loss data file')
    parser.add_argument('-o', '--output-dir', default='out')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.loss_data_file, 'r') as infile:
        data = json.load(infile)

    # Plot the training curve:

    fig = plt.figure()
    plt.plot(data['train_counter'], data['train_losses'], color='blue')
    plt.scatter(data['test_counter'], data['test_losses'], color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    ts = get_timestamp_str()
    fpath = os.path.join(args.output_dir, f'performance_graph_{ts}.png')
    print(f'saving graph of model performance to {fpath}')
    fig.savefig(fpath, dpi=fig.dpi)
