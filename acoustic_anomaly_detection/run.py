import os
import argparse
import json
import yaml
import shutil
from datetime import datetime
import mlflow
from acoustic_anomaly_detection.train import train
from acoustic_anomaly_detection.test import test
from acoustic_anomaly_detection.finetune import finetune
from acoustic_anomaly_detection.evaluate import evaluate
from acoustic_anomaly_detection.utils import (
    flatten_dict,
    update_nested_dict,
    get_params,
)


def get_run_id(run_id: str) -> str:
    all_experiments = [exp.experiment_id for exp in mlflow.search_experiments()]
    filter_string = f'attributes.`run_id` ILIKE "{run_id}%"'
    runs = mlflow.MlflowClient().search_runs(
        experiment_ids=all_experiments, filter_string=filter_string
    )
    if len(runs) == 0:
        raise ValueError(f"Run with ID {run_id} not found.")
    elif len(runs) == 1:
        return runs[0].info.run_id
    else:
        run_ids = [run.info.run_id for run in runs]
        raise ValueError(
            f"Multiple runs found with ID {run_id}. Please specify one of: {run_ids}"
        )


def create_mlflow_run(params: dict) -> str:
    experiment_name = "Test" if params["data"]["fast_dev_run"] else "Default"
    mlflow.set_experiment(experiment_name)

    run_name = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with mlflow.start_run(run_name=run_name) as mlrun:
        run_id = mlrun.info.run_id
        print(f"Started run with ID: {run_id}")

    return run_id


def start_run(args, run_id: str, params: dict):
    with mlflow.start_run(run_id=run_id):
        flatten_params = flatten_dict(params)
        mlflow.log_params(flatten_params)

    if args.train:
        train(run_id)

    if args.test:
        test(run_id)

    if args.finetune:
        finetune(run_id)

    if args.evaluate:
        evaluate(run_id)


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
    parser.add_argument(
        "--param",
        type=str,
        default="params.yaml",
        help="Path to custom hyperparameters file. Default: params.yaml",
    )
    parser.add_argument(
        "--param_variations",
        type=str,
        default=None,
        help="If specified, run all combinations of hyperparameters from this file.",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="MLflow run ID to continue logging to an existing run.",
    )

    args = parser.parse_args()

    if not os.path.exists(args.param):
        raise ValueError(f"Parameter config file {args.param} not found.")
    params = get_params(args.param)
    if args.param != "params.yaml":
        shutil.copy(args.param, "params.yaml")

    if args.all:
        args.train = True
        args.test = True
        args.finetune = True
        args.evaluate = True

    if args.run_id:
        if args.param_variations:
            raise ValueError(
                "Cannot specify both --run_id and --param_variations. Parameter combinations are only supported for new runs."
            )
        run_id = get_run_id(args.run_id)
        print(f"Resuming run with ID: {run_id}")
        start_run(args, run_id, params)
    else:
        if not args.train:
            raise ValueError(
                "Either specify --run_id to resume run or --train to start a new run."
            )
        if not args.param_variations:
            run_id = create_mlflow_run(params)
            start_run(args, run_id, params)
        else:
            if not os.path.exists(args.param_variations):
                raise ValueError(
                    f"Parameter combinations file {args.param_variations} not found."
                )
            with open(args.param_variations, "r") as f:
                param_variations = yaml.safe_load(f)

            for combination in param_variations:
                combination_params = update_nested_dict(params, combination)
                with open("params.yaml", "w") as f:
                    yaml.dump(combination_params, f)
                run_id = create_mlflow_run(combination_params)
                start_run(args, run_id, combination_params)
