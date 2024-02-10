import os
import shutil
import argparse
from datetime import datetime
import mlflow
from acoustic_anomaly_detection.train import train
from acoustic_anomaly_detection.test import test
from acoustic_anomaly_detection.finetune import finetune
from acoustic_anomaly_detection.evaluate import evaluate
from acoustic_anomaly_detection.dataset import fit_label_encoder
from acoustic_anomaly_detection.utils import (
    flatten_dict,
    update_nested_dict,
    load_params,
    save_params,
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
        fit_label_encoder()
        run_id = mlrun.info.run_id
        print(f"Started run with ID: {run_id}")

    return run_id


def copy_mlflow_run(run_dir: str, src_run_id: str, dst_run_id: str):
    client = mlflow.tracking.MlflowClient()
    src_exp_id = client.get_run(src_run_id).info.experiment_id
    src_path = os.path.join(run_dir, src_exp_id, src_run_id)
    dst_exp_id = client.get_run(dst_run_id).info.experiment_id
    dst_path = os.path.join(run_dir, dst_exp_id, dst_run_id)
    for copy_dir in ["artifacts", "metrics"]:
        shutil.copytree(
            os.path.join(src_path, copy_dir),
            os.path.join(dst_path, copy_dir),
            dirs_exist_ok=True,
        )


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


def start_variational_run(
    args,
    params: dict,
    run_id: str = None,
):
    param_variations = load_params(args.param_variations)

    for variation in param_variations:
        variation_combined = update_nested_dict(params, variation)
        save_params(variation_combined)
        variation_run_id = create_mlflow_run(variation_combined)
        if run_id:
            # if run_id is specified, copy metrics and artifacts from the specified run
            copy_mlflow_run(params["log"]["run_dir"], run_id, variation_run_id)
        start_run(args, variation_run_id, variation_combined)


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
        default=None,
        help="Path to custom hyperparameters file.",
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

    if args.all:
        args.train = True
        args.test = True
        args.finetune = True
        args.evaluate = True

    mlflow.set_tracking_uri("file:mlruns")

    if args.param_variations:
        if args.run_id:
            if args.train:
                raise ValueError(
                    "Cannot resume variational run from the training step."
                )
            else:
                # start variational run's test/finetune/evaluate steps with the config and results of the specfied run
                run_id = get_run_id(args.run_id)

                # load config from logged params
                params = load_params(args.param, run_id)
                save_params(params)
                print(f"Resuming variational run continuing from run ID: {run_id}")
                start_variational_run(args, params, run_id=run_id)
        else:
            if not args.train:
                raise ValueError(
                    "Either specify --run_id to resume variational run or --train to start a new run."
                )
            else:
                params = load_params(args.param)
                # if args.param is specified, overwrite params.yaml
                if args.param:
                    save_params(params)
                print(f"Starting variational run with params: {args.param}")
                start_variational_run(args, params)
    else:
        if args.run_id:
            run_id = get_run_id(args.run_id)
            params = load_params(args.param, run_id)
            save_params(params)
            print(f"Resuming run with ID: {run_id}")
            start_run(args, run_id, params)
        else:
            if not args.train:
                raise ValueError(
                    "Either specify --run_id to resume run or --train to start a new run."
                )
            else:
                params = load_params(args.param)
                if args.param:
                    save_params(params)
                run_id = create_mlflow_run(params)
                start_run(args, run_id, params)
