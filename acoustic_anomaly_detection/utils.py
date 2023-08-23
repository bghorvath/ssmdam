import os
import yaml
import torch

params = yaml.safe_load(open("params.yaml"))


def get_attributes(file_name: str):
    file_details = file_name.split("_")
    attributes = {k: v for k, v in zip(file_details[6::2], file_details[7::2])}
    attributes["section"] = file_details[1]
    attributes["domain"] = file_details[2]
    attributes["label"] = file_details[4]
    return attributes


def get_groupings(audio_dir: str):
    file_list = os.listdir(audio_dir)
    domains = set()
    sections = set()
    for file_name in file_list:
        section = file_name.split("_")[1]
        domain = file_name.split("_")[2]
        sections.add(section)
        domains.add(domain)
    return sections, domains


def slide_window(signal: torch.Tensor) -> torch.Tensor:
    window_size = params["transform"]["params"]["window_size"]
    stride = params["transform"]["params"]["stride"]
    batch_size, length, feature_size = signal.shape
    num_windows = (length - window_size) // stride + 1
    windows = []
    for i in range(num_windows):
        window = signal[:, i * stride : i * stride + window_size, :]
        windows.append(window)
    return torch.stack(windows, dim=1)


def reverse_slide_window(windowed_tensor: torch.Tensor) -> torch.Tensor:
    batch_size = params["train"]["batch_size"]
    num_windows = windowed_tensor.shape[0] // batch_size
    window_size = params["transform"]["params"]["window_size"]
    feature_size = windowed_tensor.shape[1] // window_size
    windowed_tensor = windowed_tensor.view(
        batch_size, num_windows, window_size, feature_size
    )
    center_idx = window_size // 2  # e.g., 2 for window size of 5

    # Take the nth value from all windows for all features
    center_values = windowed_tensor[:, :, center_idx, :]

    # Take the leftmost values from the first window for all batches and features
    left_values = windowed_tensor[:, 0, :center_idx, :]

    # Take the rightmost values from the last window for all batches and features
    right_values = windowed_tensor[:, -1, center_idx + 1 :, :]

    # Concatenate everything to reconstruct for each batch and feature
    reconstructed = torch.cat([left_values, center_values, right_values], dim=1)

    return reconstructed
