import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from utils.utils import check_dir_creation


def plot_n_heatmaps(x_train, generated_sensor_data, n=10, vmin=None, vmax=None):
    fig, axs = plt.subplots(ncols=2, nrows=n, figsize=(10, 5 * n), sharey=True)
    if n > 1:
        for i in range(n):
            sns.heatmap(x_train[i], center=0, ax=axs[i, 0], vmin=x_train[:n].min(), vmax=x_train[:n].max(),
                        xticklabels=['x', 'y', 'z'])
            sns.heatmap(generated_sensor_data[i], center=0, ax=axs[i, 1], vmin=x_train[:n].min(),
                        vmax=x_train[:n].max(), xticklabels=['x', 'y', 'z'])
    else:
        for i in range(n):
            sns.heatmap(x_train[i], center=0, ax=axs[0], vmin=x_train[:n].min(), vmax=x_train[:n].max(),
                        xticklabels=['x', 'y', 'z'])
            sns.heatmap(generated_sensor_data[i], center=0, ax=axs[1], vmin=x_train[:n].min(),
                        vmax=x_train[:n].max(), xticklabels=['x', 'y', 'z'])

    return fig


def plot_n_lineplots(x_train, generated_sensor_data, n=10, vmin=None, vmax=None):
    x_train_t = np.array([np.transpose(ts) for ts in x_train[:n]])
    generated_sensor_data_t = np.array([np.transpose(ts) for ts in generated_sensor_data[:n]])
    time_steps = len(x_train_t[0][0])

    # plt.ylim(-.25, .25)
    fig, axs = plt.subplots(ncols=2, nrows=n, figsize=(10, 5 * n), sharey=True)
    if n > 1:
        for i in range(n):
            ax1 = sns.lineplot(np.arange(time_steps), x_train_t[i][0], ax=axs[i, 0])
            ax1 = sns.lineplot(np.arange(time_steps), x_train_t[i][1], ax=axs[i, 0])
            ax1 = sns.lineplot(np.arange(time_steps), x_train_t[i][2], ax=axs[i, 0])
            # ax1.legend(['x', 'y', 'z'])

            ax2 = sns.lineplot(np.arange(time_steps), generated_sensor_data_t[i][0], ax=axs[i, 1])
            ax2 = sns.lineplot(np.arange(time_steps), generated_sensor_data_t[i][1], ax=axs[i, 1])
            ax2 = sns.lineplot(np.arange(time_steps), generated_sensor_data_t[i][2], ax=axs[i, 1])
            ax2.legend(['x', 'y', 'z'])
    else:
        for i in range(n):
            ax1 = sns.lineplot(np.arange(time_steps), x_train_t[i][0], ax=axs[0])
            ax1 = sns.lineplot(np.arange(time_steps), x_train_t[i][1], ax=axs[0])
            ax1 = sns.lineplot(np.arange(time_steps), x_train_t[i][2], ax=axs[0])
            # ax1.legend(['x', 'y', 'z'])

            ax2 = sns.lineplot(np.arange(time_steps), generated_sensor_data_t[i][0], ax=axs[1])
            ax2 = sns.lineplot(np.arange(time_steps), generated_sensor_data_t[i][1], ax=axs[1])
            ax2 = sns.lineplot(np.arange(time_steps), generated_sensor_data_t[i][2], ax=axs[1])
            ax2.legend(['x', 'y', 'z'])

    return fig
