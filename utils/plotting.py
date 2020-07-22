import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from utils.utils import check_dir_creation


def plot_n_heatmaps(x_train, generated_sensor_data, n=10, vmin=None, vmax=None, save_dir=None, file_name='plot_heatmap'):
    
    fig, axs = plt.subplots(ncols=2, nrows=n, figsize=(10, 5*n), sharey=True)
    for i in range(n):
        sns.heatmap(x_train[i], center=0, ax=axs[i, 0], vmin=x_train[:10].min(), vmax=x_train[:10].max())
        sns.heatmap(generated_sensor_data[i], center=0, ax=axs[i, 1], vmin=x_train[:10].min(), vmax=x_train[:10].max())

    if save_dir:
        check_dir_creation(save_dir)
        plt.savefig(os.path.join(save_dir, file_name +str(i)))


def plot_n_lineplots(x_train, generated_sensor_data, n=10, vmin=None, vmax=None, save_dir=None, file_name='plot_line'):
    x_train_t = np.array([np.transpose(ts) for ts in x_train[:n]])
    generated_sensor_data_t = np.array([np.transpose(ts) for ts in generated_sensor_data[:n]])
    time_steps = len(x_train_t[0][0])
    
    # plt.ylim(-.25, .25)
    fig, axs = plt.subplots(ncols=2, nrows=n, figsize=(10, 5*n), sharey=True)
    for i in range(n):

        sns.lineplot(np.arange(time_steps), x_train_t[i][0], ax=axs[i, 0])
        sns.lineplot(np.arange(time_steps), x_train_t[i][1], ax=axs[i, 0])
        sns.lineplot(np.arange(time_steps), x_train_t[i][2], ax=axs[i, 0])

        sns.lineplot(np.arange(time_steps), generated_sensor_data_t[i][0], ax=axs[i, 1])
        sns.lineplot(np.arange(time_steps), generated_sensor_data_t[i][1], ax=axs[i, 1])
        sns.lineplot(np.arange(time_steps), generated_sensor_data_t[i][2], ax=axs[i, 1])

    if save_dir:
        check_dir_creation(save_dir)
        plt.savefig(os.path.join(save_dir, file_name + str(i)))