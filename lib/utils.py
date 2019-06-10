import json
import os
from typing import List


def save_loss_data_file(
        output_dir: str,
        train_counter: List[int],
        train_losses: List[float],
        test_counter: List[int],
        test_losses: List[float]) -> None:

    data = {
        'train_counter': train_counter,
        'train_losses': train_losses,
        'test_counter': test_counter,
        'test_losses': test_losses,
    }
    loss_data_file = os.path.join(output_dir, 'loss_data.json')
    with open(loss_data_file, 'w') as outfile:
        json.dump(data, outfile, indent=2)
