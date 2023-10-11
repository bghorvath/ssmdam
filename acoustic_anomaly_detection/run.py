import argparse
import json
from datetime import datetime
import mlflow
from acoustic_anomaly_detection.train import train
from acoustic_anomaly_detection.test import test
from acoustic_anomaly_detection.utils import init_train_status, get_train_status


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run training/testing/finetuning/evaluation steps."
    )
    parser.add_argument("--train", action="store_true", help="Run the training step.")
    parser.add_argument("--test", action="store_true", help="Run the testing step.")
    # parser.add_argument("--finetune", action="store_true", help="Run the finetuning step.")
    # parser.add_argument("--eval", action="store_true", help="Run the evaluation step.")
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="MLflow run ID to continue logging to an existing run.",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="Default",
        help="Name of the MLflow experiment.",
    )

    args = parser.parse_args()

    mlflow.set_experiment(args.experiment_name)

    if not args.run_id:
        if not args.train:
            raise ValueError(
                "If not resuming an existing run, you must specify the --train flag."
            )

        run_name = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with mlflow.start_run(run_name=run_name):
            train_status = init_train_status()
            mlflow.set_tag("train_status", json.dumps(train_status))
            run_id = mlflow.active_run().info.run_id
            print(f"Started run with ID: {run_id}")
    else:
        run_id = args.run_id
        if mlflow.get_run(run_id) is None:
            raise ValueError(f"Run with ID {run_id} not found.")
        print(f"Resuming run with ID: {run_id}")
        train_status = get_train_status(run_id)

    if args.train:
        to_train = ", ".join(
            [
                data_path.split("/")[-1]
                for data_path, status in train_status["trained"].items()
                if not status
            ]
        )
        print(f"Training: {to_train}")
        train(run_id)

    if args.test:
        to_test = ", ".join(
            [
                data_path.split("/")[-1]
                for data_path, status in train_status["tested"].items()
                if not status
            ]
        )
        print(f"Testing: {to_test}")
        test(run_id)
