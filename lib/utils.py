import datetime
import json
import os

from typing import List


def save_loss_data_file(
        output_dir: str,
        train_counter: List[int],
        train_losses: List[float],
        test_counter: List[int],
        test_losses: List[float],
        n_epochs: int) -> None:

    data = {
        'train_counter': train_counter,
        'train_losses': train_losses,
        'test_counter': test_counter,
        'test_losses': test_losses,
        'epochs': n_epochs
    }
    ts = get_timestamp_str()
    loss_data_file = os.path.join(output_dir, f'loss_data_{ts}.json')
    print(f'saving loss data to {loss_data_file}')
    with open(loss_data_file, 'w') as outfile:
        json.dump(data, outfile, indent=2)


def get_timestamp_str() -> str:
    return datetime.datetime.now().strftime('%Y%m%d_T%H:%M:%S')
