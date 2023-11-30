import os
import ast
import yaml
import json
import sys
import numpy as np
import pandas as pd
from scipy import stats
import torch
from torchmetrics.functional.classification import (
    binary_auroc,
    binary_precision,
    binary_recall,
    binary_f1_score,
)
import mlflow


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


def slice_signal(signal: torch.Tensor, window_size: int, stride: int) -> torch.Tensor:
    """
    Splits a tensor into windows of a specified size and stride.
    Expected shape: (batch_size, length, feature_size)
    Returns a tensor of shape (batch_size, num_windows, window_size, feature_size)
    """
    batch_size, length, feature_size = signal.shape
    num_windows = (length - window_size) // stride + 1
    windows = []
    for i in range(num_windows):
        window = signal[:, i * stride : i * stride + window_size, :]
        windows.append(window)
    return torch.stack(windows, dim=1)


def reconstruct_signal(sliced_tensor: torch.Tensor, batch_size: int, window_size: int):
    """
    Reconstructs the original tensor from a windowed tensor.
    Expected shape: (batch_size, num_windows, window_size, feature_size)
    Returns a tensor of shape (batch_size, length, feature_size)
    """
    num_windows = sliced_tensor.shape[0] // batch_size
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


def calculate_metrics(
    error_score: torch.Tensor,
    y_true: torch.Tensor,
    max_fpr: float,
    decision_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    auc = binary_auroc(error_score, y_true)
    p_auc = binary_auroc(error_score, y_true, max_fpr=max_fpr)
    prec = binary_precision(error_score, y_true, threshold=decision_threshold)
    recall = binary_recall(error_score, y_true, threshold=decision_threshold)
    f1 = binary_f1_score(error_score, y_true, threshold=decision_threshold)
    return auc, p_auc, prec, recall, f1


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


def flatten_dict(d: dict) -> dict:
    """
    Flattens a nested dictionary.
    Example:
    d: {"a": {"b": 1, "c": {"d": [2, 3]}}}
    Returns: {"a.b": 1, "a.c.d": [2, 3]}
    """

    def expand(key, value):
        if isinstance(value, dict):
            return [(key + "." + k, v) for k, v in flatten_dict(value).items()]
        else:
            return [(key, value)]

    items = [item for k, v in d.items() for item in expand(k, v)]
    return dict(items)


def unflatten_dict(d: dict) -> dict:
    """
    Unflattens a flattened dictionary.
    Example:
    d: {"a.b": 1, "a.c.d": [2, 3]}
    Returns: {"a": {"b": 1, "c": {"d": [2, 3]}}}
    """
    result = {}
    for key, value in d.items():
        parts = key.split(".")
        d = result
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = value
    return result


def update_nested_dict(nested_dict: dict, flattened_dict: dict) -> dict:
    """
    Update keys in nested dictionary d with values from u if they exist.
    Example:
    d: {"a": {"b": 1, "c": 2}}
    u: {"a.b": 3, "a.d": 4}
    """
    for k, v in flattened_dict.items():
        keys = k.split(".")
        if len(keys) == 1:
            nested_dict[k] = v
        else:
            nested_dict[keys[0]] = update_nested_dict(
                nested_dict[keys[0]], {k[len(keys[0]) + 1 :]: v}
            )
    return nested_dict


def convert_to_original_format(value: str):
    """
    Converts a string to its original format. Needed for loading parameters from MLflow.
    """
    try:
        return ast.literal_eval(value)
    except:
        return value


def load_params(params_file: str = None, run_id: str = None) -> dict:
    """
    Returns a dictionary of parameters.
    If params_file is specified, it loads the parameters from the file.
    If run_id is specified, it loads the parameters from the MLflow run.
    """
    if params_file and run_id:
        raise ValueError("Please specify only one of params_file and run_id.")

    if run_id:
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(run_id)
        params = run.data.params
        converted_params = {k: convert_to_original_format(v) for k, v in params.items()}
        unflatten_params = unflatten_dict(converted_params)
        return unflatten_params

    params_file = params_file or "params.yaml"
    if not os.path.exists(params_file):
        raise ValueError(f"Parameter config file {params_file} not found.")

    with open(params_file, "r") as f:
        params = yaml.safe_load(f)
    return params


def save_params(params: dict):
    with open("params.yaml", "w") as f:
        yaml.dump(params, f)
