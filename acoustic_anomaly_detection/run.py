import os
import yaml
from tqdm import tqdm
from train import train
from evaluate import evaluate

params = yaml.safe_load(open("params.yaml"))

if __name__ == "__main__":
    audio_dirs = params["data"]["audio_dirs"]
    for audio_dir in tqdm(audio_dirs):
        machine_type = audio_dir.split("/")[-1]
        files = os.listdir(f"{audio_dir}/train")
        machine_ids = {file.split("_")[2] for file in files}
        for machine_id in tqdm(machine_ids):
            train(machine_type, machine_id)
            evaluate(machine_type, machine_id)
