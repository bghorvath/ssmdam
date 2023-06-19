from dataset import AudioDataset
from model import get_model
from torch.utils.data import DataLoader
from lightning import Trainer
from dvclive import Live
from dvclive.lightning import DVCLiveLogger
import yaml

params = yaml.safe_load(open("params.yaml"))

def evaluate():
    num_workers = params["misc"]["num_workers"]
    log_dir = params["misc"]["log_dir"]
    ckpt_path = params["misc"]["ckpt_path"]

    dataset = AudioDataset(test=True, fast_dev_run=params["data"]["fast_dev_run"])
    test_loader = DataLoader(dataset, batch_size=128, num_workers=num_workers, shuffle=False)


    input_size = dataset[0][0].shape[1:].numel()
    model = get_model(input_size=input_size)
    model = model.load_from_checkpoint(ckpt_path)

    with Live(
        dir=log_dir,
        resume=True,
        ) as live:
        trainer = Trainer(
            logger=DVCLiveLogger(experiment=live)
        )
        trainer.test(
            model,
            dataloaders=test_loader,
            ckpt_path=ckpt_path,
        )

if __name__ == "__main__":
    evaluate()
