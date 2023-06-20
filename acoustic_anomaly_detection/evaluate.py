import yaml
from dataset import AudioDataset
from model import get_model
from torch.utils.data import DataLoader
from lightning import Trainer
from dvclive import Live
from dvclive.lightning import DVCLiveLogger

params = yaml.safe_load(open("params.yaml"))


def evaluate(
    machine_type: str,
    machine_id: str,
):
    num_workers = params["misc"]["num_workers"]
    log_dir = params["misc"]["log_dir"]
    ckpt_path = params["misc"]["ckpt_path"]

    dataset = AudioDataset(
        machine_type=machine_type,
        machine_id=machine_id,
        test=True,
        fast_dev_run=params["data"]["fast_dev_run"],
    )
    test_loader = DataLoader(
        dataset, batch_size=128, num_workers=num_workers, shuffle=False
    )

    input_size = dataset[0][0].shape[1:].numel()
    model = get_model(input_size=input_size)
    model = model.load_from_checkpoint(ckpt_path)

    exp_name = f"{machine_type}_{machine_id}"

    with Live(
        dir=log_dir,
        resume=True,
        save_dvc_exp=True,
    ) as live:
        live._exp_name = exp_name
        trainer = Trainer(logger=DVCLiveLogger(experiment=live))
        trainer.test(
            model,
            dataloaders=test_loader,
            ckpt_path=ckpt_path,
        )
