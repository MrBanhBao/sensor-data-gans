import pandas as pd
import numpy as np
from typing import List, Dict


def get_random_windows(series: pd.Series, window_size: int, sample_size: int, col_names: List[str]) -> List[pd.Series]:
    sampled_window_series: List[pd.Series] = []
    time_steps: int = len(series[col_names[0]])
    for i in range(sample_size):
        if window_size > time_steps:
            continue
        else:
            start_window_idx: int = np.random.randint(low=0, high=time_steps-window_size)
            new_series: pd.Series = series.copy()
            for col_name in col_names:
                arr: List[float] = series[col_name][start_window_idx:start_window_idx+window_size]
                new_series.set_value(col_name, arr)
            sampled_window_series.append(new_series)
    return sampled_window_series


def get_sliding_windows(series: pd.Series, window_size: int, step_size: int, col_names: List[str]) -> List[pd.Series]:
    sliding_window_series: List[pd.Series] = []
    time_steps: int = len(series[col_names[0]])
    iterations: int = int((time_steps-window_size)/step_size+1)
    start_window_idx: int = 0

    for i in range(iterations):
        if window_size > time_steps:
            continue
        else:
            new_series: pd.Series = series.copy()
            for col_name in col_names:
                arr: List[float] = series[col_name][start_window_idx:start_window_idx+window_size]
                new_series.at[col_name] = arr
            start_window_idx += step_size
            sliding_window_series.append(new_series)
    return sliding_window_series


def windowing_dataframe(df: pd.DataFrame, window_size: int, step_or_sample_size: int, col_names: List[str],
                        method: str = 'sliding') -> pd.DataFrame:
    window_series = []
    for i, series in df.iterrows():
        if method == 'sliding':
            window_series += get_sliding_windows(series=series, window_size=window_size,
                                                 step_size=step_or_sample_size, col_names=col_names)
        elif method == 'random':
            window_series += get_random_windows(series=series, window_size=window_size,
                                                sample_size=step_or_sample_size, col_names=col_names)

    return pd.DataFrame(window_series)


def transform_windows_df(windowed_df: pd.DataFrame, input_cols: List[str], one_hot_encode=True,
                         as_channel: bool = False):
    x = []
    y = []
    num_classes = len(windowed_df.act.unique())
    
    for i, window in windowed_df.iterrows():
        col_stack = []
        for col in input_cols:
            col_stack.append(window[col])

        if as_channel:
            x.append(np.array(col_stack).reshape(np.array(col_stack).shape[1], 1, np.array(col_stack).shape[0]))
        else:
            x.append(np.array(col_stack).transpose())

        if one_hot_encode:
            one_hot_vec = np.zeros(num_classes)
            one_hot_vec[int(window['act'])] = 1
            y.append(one_hot_vec)
        else:
            y.append(window['act'])

    return np.array(x), np.array(y)

