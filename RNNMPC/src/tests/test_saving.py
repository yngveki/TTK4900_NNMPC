import csv
from pathlib import Path

def append_test_mse(csv_path, model_name, test_name, mse, **hyperparameters):
    assert isinstance(csv_path, Path), 'Path given must be of the \'Path\' type!'
    row = [model_name, test_name, mse]

    if hyperparameters:
        for param in hyperparameters.values():
            row.append(param)
    
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)

        writer.writerow(row)

if __name__ == '__main__':
    csv_path = Path(__file__).parent / '../../models/test_saving.csv'
    model_name = 'model_grid_0'
    test_name = 'random_walk_10k_globally_normalized'
    mse = 0.0000059066
    append_test_mse(csv_path, model_name, test_name, mse, m=30)

    print('finished!')