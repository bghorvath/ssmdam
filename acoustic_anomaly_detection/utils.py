import os
import yaml
import torch
import json
import mlflow

params = yaml.safe_load(open("params.yaml"))


def get_attributes(file_path: str):
    """
    Extracts attributes from a file path.
    Expected format: data/{data_source}/{dev_eval}/{machine_type}/{train_test}/{file_name}
    file_name: section_{section}_{domain}_{split}_{label}_{attribute1}_{value1}_{attribute2}_{value2}...
    """
    _, data_source, dev_eval, machine_type, train_test, file_name = file_path.split("/")
    attributes["data_source"] = data_source
    attributes["dev_eval"] = dev_eval
    attributes["machine_type"] = machine_type
    attributes["train_test"] = train_test

    file_details = file_name.split("_")
    attributes["section"] = file_details[1]

    # for validation data
    if len(file_details) <= 3:
        return attributes

    attributes = {k: v for k, v in zip(file_details[6::2], file_details[7::2])}
    attributes["domain"] = file_details[2]
    # attributes["split"] = file_details[3]
    attributes["label"] = file_details[4]
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


def reconstruct_signal(sliced_tensor: torch.Tensor) -> torch.Tensor:
    """
    Reconstructs the original tensor from a windowed tensor.
    Expected shape: (batch_size, num_windows, window_size, feature_size)
    Returns a tensor of shape (batch_size, length, feature_size)
    """
    batch_size = params["train"]["batch_size"]
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


params = yaml.safe_load(open("params.yaml"))


def init_train_status() -> dict[str, dict[str, bool]]:
    """
    Initialize a dictionary with models and their train status.
    """
    data_sources = params["data"]["data_sources"]
    dev_data_paths = [
        os.path.join("data", data_source, "dev", data_dir)
        for data_source in data_sources
        for data_dir in os.listdir(os.path.join("data", data_source, "dev"))
    ]
    eval_data_paths = [
        os.path.join("data", data_source, "eval", data_dir)
        for data_source in data_sources
        for data_dir in os.listdir(os.path.join("data", data_source, "eval"))
    ]
    return {
        "trained": {data_path: False for data_path in dev_data_paths},
        "tested": {data_path: False for data_path in dev_data_paths},
        "finetuned": {data_path: False for data_path in eval_data_paths},
        "evaluated": {data_path: False for data_path in eval_data_paths},
    }


def get_train_status(run_id: str) -> dict[str, dict[str, bool]]:
    """
    Fetch the training status dictionary from MLflow.
    """
    status_str = mlflow.get_run(run_id).data.tags.get("train_status", "{}")
    return json.loads(status_str)


def update_train_status(
    run_id: str, stage: str, model_name: str, step: str, status: bool = True
) -> None:
    """
    Update the train status of a model for a specific step in MLflow.
    """
    current_status = get_train_status(run_id)
    current_status[stage][model_name][step] = status
    mlflow.set_tag("train_status", json.dumps(current_status))
