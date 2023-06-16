from dataset import AudioDataset
from model import LitAutoEncoder
from torch.utils.data import DataLoader
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from dvclive import Live
from dvclive.lightning import DVCLiveLogger
import yaml

params = yaml.safe_load(open("params.yaml"))

def evaluate():
    ckpt_path = params["train"]["ckpt_path"]
    num_workers = params["train"]["num_workers"]

    dataset = AudioDataset(test=True)
    test_loader = DataLoader(dataset, batch_size=128, num_workers=num_workers, shuffle=False)

    model = LitAutoEncoder.load_from_checkpoint(ckpt_path)

    trainer = Trainer(
        logger=DVCLiveLogger()
        )
    trainer.test(
        model,
        dataloaders=test_loader,
        ckpt_path=ckpt_path,
    )

if __name__ == "__main__":
    evaluate()
