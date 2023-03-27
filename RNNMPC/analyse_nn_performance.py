import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

def plot_mse_specific_test(n_models, test_name, **kwargs):
    x_axis = np.linspace(0, n_models - 1, n_models)
    # -- PLOTTING -- #
    fig, ax = plt.subplots(1,1)
    ax.set_title(f'MSE when testing model on different test sets')
    for test_name, mse in kwargs.items():
        ax.plot(x_axis, mse, '-', label=f'MSE, \'{test_name}\'', linewidth=2.0)
    ax.set_ylabel('MSE as function of model number')
    ax.legend(loc='best', prop={'size': 15})
    ax.set_yscale('log')


if __name__ == '__main__':
    csv_path = Path(__file__).parent / 'models/model_grid_second_mses.csv'
    df = pd.read_csv(csv_path)

    test_names = set(df['test_name'].tolist())

    mses = {}
    for test_name in test_names:
        relevant_rows = df[df['test_name'] == test_name]

        mses[test_name] = relevant_rows['mse'].tolist()

        # Deprecated, but kept for future expansion (want to plot hyperparameters, maybe)
        cols = {}
        for key in df.head():
            cols[key] = relevant_rows[key].tolist()
        
    plot_mse_specific_test(len(cols['model_name']), test_name, **mses)

    plt.show()