import os
import yaml
from tqdm import tqdm
from torch.utils.data import DataLoader
from lightning import Trainer
import mlflow
from model import get_model
from dataset import AudioDataset
from utils import get_runs_scheduled, get_previous_runs

params = yaml.safe_load(open("params.yaml"))


def evaluate_run(
    machine_type: str,
    machine_id: str,
    run_id: str,
):
    log_dir = params["misc"]["log_dir"]
    batch_size = params["train"]["batch_size"]
    num_workers = params["misc"]["num_workers"]

    test_dataset = AudioDataset(
        machine_type=machine_type,
        machine_id=machine_id,
        test=True,
        fast_dev_run=params["data"]["fast_dev_run"],
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        drop_last=True,
    )

    input_size = test_dataset[0][0].shape[1:].numel()
    model = get_model(input_size=input_size)

    ckpt_path = os.path.join(log_dir, run_id, "checkpoints", "best.ckpt")
    model = model.load_from_checkpoint(ckpt_path)

    exp_name = f"{machine_type}_{machine_id}"
    trainer = Trainer(
        logger=mlflow.pytorch.MLFlowLogger(
            experiment_name=exp_name,
            tracking_uri=log_dir,
        )
    )
    trainer.test(
        model,
        dataloaders=test_loader,
        ckpt_path=ckpt_path,
    )


def evaluate():
    log_dir = params["misc"]["log_dir"]

    runs_scheduled = get_runs_scheduled()

    # only evaluate runs that have not been evaluated yet but have been completed
    completed_runs, incomplete_runs, parent_run_id = get_previous_runs()
    runs_scheduled = {
        k: v for k, v in runs_scheduled.items() if k not in completed_runs
    }
    runs_scheduled.update(incomplete_runs)

    if len(runs_scheduled) == 0:
        print("No new runs to evaluate")
        return

    mlflow.set_tracking_uri(log_dir)
    with mlflow.start_run(run_id=parent_run_id) as run:
        for (machine_type, machine_id), child_run_id in tqdm(runs_scheduled.items()):
            with mlflow.start_run(
                run_name=f"{machine_type}_{machine_id}",
                run_id=child_run_id,
                nested=True,
            ) as child_run:
                evaluate_run(machine_type, machine_id, run_id=child_run_id)
