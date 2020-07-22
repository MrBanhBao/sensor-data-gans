import numpy as np
import pandas as pd


def calc_consultant(df, xyz=['userAcceleration.x','userAcceleration.y','userAcceleration.z']):
    cs = []
    for i, row in df.iterrows():
        x = np.array(row[xyz[0]])
        y = np.array(row[xyz[1]])
        z = np.array(row[xyz[2]])
        c = np.square(np.power(x, 2) + np.power(y, 2) + np.power(z, 2))
        cs.append(c)
    return cs


def filter_by_activity_index(x: np.array, y: np.array, activity_idx: int) -> np.array:
    activity_mask = y == activity_idx
    x_activity = x[activity_mask]
    y_activity = y[activity_mask]

    if x_activity.shape[-1] == 1:
        return x_activity.reshape(x_activity.shape[0], x_activity.shape[1]), y_activity
    else:
        return x_activity, y_activity
