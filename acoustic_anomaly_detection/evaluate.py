import os
import yaml
from tqdm import tqdm
from dataset import AudioDataset
from model import get_model
from torch.utils.data import DataLoader
from lightning import Trainer
from dvclive import Live
from dvclive.lightning import DVCLiveLogger

from dataset import AudioDataset
from model import get_model

params = yaml.safe_load(open("params.yaml"))


def evaluate():
    num_workers = params["misc"]["num_workers"]
    log_dir = params["misc"]["log_dir"]
    audio_dirs = params["data"]["audio_dirs"]
    ckpt_dir = params["misc"]["ckpt_dir"]
    fast_dev_run = params["data"]["fast_dev_run"]

    with Live(
        dir=log_dir,
        resume=True,
        # save_dvc_exp=True,
    ) as live:
        #         live._exp_name = exp_name
        for audio_dir in tqdm(audio_dirs):
            machine_type = audio_dir.split("/")[-1]
            audio_dir = os.path.join(audio_dir, "test")

            machine_ids = {p.split("_")[2] for p in os.listdir(audio_dir)}

            ckpt_path = os.path.join(ckpt_dir, machine_type + ".ckpt")

            for machine_id in machine_ids:
                file_list = [
                    os.path.join(audio_dir, file)
                    for file in os.listdir(audio_dir)
                    if file.split("_")[2] == machine_id
                ]

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
                model.model_name = f"{machine_type}_{machine_id}"  # type: ignore

                logger = DVCLiveLogger(
                    experiment=live,
                    save_dvc_exp=False,
                )

                trainer = Trainer(logger=logger)

                trainer.test(
                    model=model,  # type: ignore
                    dataloaders=test_loader,
                    # ckpt_path=ckpt_path,
                )


if __name__ == "__main__":
    evaluate()
