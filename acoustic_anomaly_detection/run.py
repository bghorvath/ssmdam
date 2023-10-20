import argparse
import json
import yaml
from datetime import datetime
import mlflow
from acoustic_anomaly_detection.train import train
from acoustic_anomaly_detection.test import test

params = yaml.safe_load(open("params.yaml"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run training/testing/finetuning/evaluation steps."
    )
    parser.add_argument("--all", action="store_true", help="Run all steps.")
    parser.add_argument("--train", action="store_true", help="Run the training step.")
    parser.add_argument("--test", action="store_true", help="Run the testing step.")
    parser.add_argument(
        "--finetune", action="store_true", help="Run the finetuning step."
    )
    parser.add_argument(
        "--evaluate", action="store_true", help="Run the evaluation step."
    )
    # parser.add_argument("--config", type=str, default=None, help="Path to config file."")
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="MLflow run ID to continue logging to an existing run.",
    )

    args = parser.parse_args()

    if args.all:
        args.train = True
        args.test = True
        args.finetune = True
        args.evaluate = True

    if not args.run_id:
        experiment_name = "Test" if params["data"]["fast_dev_run"] else "Default"
        mlflow.set_experiment(experiment_name)
        if not args.train:
            raise ValueError(
                "Either specify run_id to resume run, or specify --train to start a new run."
            )

        run_name = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with mlflow.start_run(run_name=run_name) as mlrun:
            run_id = mlrun.info.run_id
            print(f"Started run with ID: {run_id}")
    else:
        run_id = args.run_id
        if mlflow.get_run(run_id) is None:
            raise ValueError(f"Run with ID {run_id} not found.")
        print(f"Resuming run with ID: {run_id}")

    if args.train:
        train(run_id)

    if args.test:
        test(run_id)

    # if args.finetune:
    #     finetune(run_id)

    # if args.evaluate:
    #     evaluate(run_id)
