import os
import yaml
from tqdm import tqdm
from dataset import AudioDataset
from model import get_model
from torch.utils.data import DataLoader
from lightning import Trainer
from dvclive import Live
from dvclive.lightning import DVCLiveLogger

from acoustic_anomaly_detection.dataset import AudioDataset
from acoustic_anomaly_detection.model import get_model
from acoustic_anomaly_detection.utils import get_groupings

params = yaml.safe_load(open("params.yaml"))


def test():
    num_workers = params["misc"]["num_workers"]
    log_dir = params["misc"]["log_dir"]
    data_sources = params["data"]["data_sources"]
    ckpt_dir = params["misc"]["ckpt_dir"]
    fast_dev_run = params["data"]["fast_dev_run"]

    with Live(
        dir=log_dir,
        resume=True,
        # save_dvc_exp=True,
    ) as live:
        #         live._exp_name = exp_name
        for i, data_source in enumerate(tqdm(data_sources)):
            print(f"Testing ({i+1}/{len(data_sources)} data source: {data_source})")
            audio_dirs_path = os.path.join("data", "prepared", data_source, "dev")
            audio_dirs = [
                os.path.join(audio_dirs_path, dir)
                for dir in os.listdir(audio_dirs_path)
            ]
            for j, audio_dir in enumerate(tqdm(audio_dirs)):
                machine_type = audio_dir.split("/")[-1]
                print(f"Testing ({j+1}/{len(audio_dirs)} machine type: {machine_type})")
                audio_dir = os.path.join(audio_dir, "test")

                ckpt_path = os.path.join(ckpt_dir, machine_type + ".ckpt")
                if not os.path.exists(ckpt_path):
                    print(f"Model for {machine_type} not found. Skipping...")
                    continue

                if not os.path.exists(audio_dir):
                    print(f"Test data for {machine_type} not found. Skipping...")
                    continue

                sections, domains = get_groupings(audio_dir)

                for domain in domains:
                    file_list = [
                        os.path.join(audio_dir, file)
                        for file in os.listdir(audio_dir)
                        if file.split("_")[2] == domain
                    ]

                    if len(file_list) == 0:
                        print(f"No data for {machine_type} found. Skipping...")
                        continue

                    dataset = AudioDataset(
                        file_list=file_list,
                        fast_dev_run=fast_dev_run,
                    )

                    test_loader = DataLoader(
                        dataset,
                        batch_size=1,
                        num_workers=num_workers,
                        shuffle=False,
                        drop_last=True,
                    )

                    input_size = dataset[0][0].shape[1:].numel()

                    model = get_model(model_name="", input_size=input_size)
                    model = model.load_from_checkpoint(ckpt_path)
                    model.model_name = f"{machine_type}_{domain}"

                    trainer = Trainer(logger=DVCLiveLogger(experiment=live))

                    trainer.test(
                        model=model,  # type: ignore
                        dataloaders=test_loader,
                        # ckpt_path=ckpt_path,
                    )


if __name__ == "__main__":
    test()
