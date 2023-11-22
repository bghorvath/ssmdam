import os
import yaml
import json
import sys
import numpy as np
import pandas as pd
from scipy import stats
import torch
import mlflow

params = yaml.safe_load(open("params.yaml"))


def get_attributes(file_path: str):
    """
    Extracts attributes from a file path.
    Expected format: data/{data_source}/{dev_eval}/{machine_type}/{train_test}/{file_name}
    file_name: section_{section}_{domain}_{split}_{label}_{attribute1}_{value1}_{attribute2}_{value2}...
    """
    _, data_source, dev_eval, machine_type, train_test, file_name = file_path.replace(
        ".wav", ""
    ).split("/")
    attributes = {}
    attributes["data_source"] = data_source
    attributes["dev_eval"] = dev_eval
    attributes["machine_type"] = machine_type
    attributes["train_test"] = train_test

    file_details = file_name.split("_")
    attributes["section"] = file_details[1]

    # for validation data
    if len(file_details) <= 3:
        return attributes

    attributes["domain"] = file_details[2]
    # attributes["split"] = file_details[3]
    attributes["label"] = file_details[4]
    add_attributes = {k: v for k, v in zip(file_details[6::2], file_details[7::2])}
    attributes.update(add_attributes)
    return attributes


def slice_signal(signal: torch.Tensor) -> torch.Tensor:
    """
    Splits a tensor into windows of a specified size and stride.
    Expected shape: (batch_size, length, feature_size)
    Returns a tensor of shape (batch_size, num_windows, window_size, feature_size)
    """
    window_size = params["transform"]["params"]["window_size"]
    stride = params["transform"]["params"]["stride"]
    batch_size, length, feature_size = signal.shape
    num_windows = (length - window_size) // stride + 1
    windows = []
    for i in range(num_windows):
        window = signal[:, i * stride : i * stride + window_size, :]
        windows.append(window)
    return torch.stack(windows, dim=1)


def reconstruct_signal(sliced_tensor: torch.Tensor, batch_size: int) -> torch.Tensor:
    """
    Reconstructs the original tensor from a windowed tensor.
    Expected shape: (batch_size, num_windows, window_size, feature_size)
    Returns a tensor of shape (batch_size, length, feature_size)
    """
    num_windows = sliced_tensor.shape[0] // batch_size
    window_size = params["transform"]["params"]["window_size"]
    feature_size = sliced_tensor.shape[1] // window_size
    sliced_tensor = sliced_tensor.view(
        batch_size, num_windows, window_size, feature_size
    )
    center_idx = window_size // 2  # e.g., 2 for window size of 5

    # Take the nth value from all windows for all features
    center_values = sliced_tensor[:, :, center_idx, :]

    # Take the leftmost values from the first window for all batches and features
    left_values = sliced_tensor[:, 0, :center_idx, :]

    # Take the rightmost values from the last window for all batches and features
    right_values = sliced_tensor[:, -1, center_idx + 1 :, :]

    # Concatenate everything to reconstruct for each batch and feature
    reconstructed = torch.cat([left_values, center_values, right_values], dim=1)

    return reconstructed


def save_metrics(metrics_dict: dict, artifacts_dir: str, stage: str):
    """
    Saves metrics to a JSON file.
    """
    metrics_array = np.array(list(metrics_dict.values()))
    metrics_dict["Arithmetic mean"] = np.mean(metrics_array, axis=0)
    metrics_dict["Harmonic mean"] = stats.hmean(
        np.maximum(metrics_array, sys.float_info.epsilon), axis=0
    )

    df_cols = (
        ["AUC", "pAUC", "Precision", "Recall", "F1"]
        + [
            "AUC (source)",
            "pAUC (source)",
            "Precision (source)",
            "Recall (source)",
            "F1 (source)",
        ]
        + [
            "AUC (target)",
            "pAUC (target)",
            "Precision (target)",
            "Recall (target)",
            "F1 (target)",
        ]
    )

    metrics_df = pd.DataFrame.from_dict(metrics_dict, orient="index", columns=df_cols)
    metrics_df.index.name = "Machine type"
    metrics_df.reset_index(inplace=True)
    metrics_df.to_csv(os.path.join(artifacts_dir, f"{stage}_metrics.csv"), index=False)
