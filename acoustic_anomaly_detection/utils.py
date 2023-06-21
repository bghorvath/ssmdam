import os
import yaml
import mlflow

params = yaml.safe_load(open("params.yaml"))


def get_runs_scheduled() -> dict[tuple[str, str], str]:
    data_dir = params["data"]["data_dir"]
    machine_types = params["data"]["source_dirs"]

    runs_scheduled = {}
    for machine_type in machine_types:
        files_path = os.path.join(data_dir, machine_type, "test")
        files = os.listdir(files_path)
        machine_ids = {file.split("_")[2] for file in files}
        runs_scheduled.update(
            {(machine_type, machine_id): None for machine_id in machine_ids}
        )
    return runs_scheduled


def get_previous_runs() -> tuple[list[tuple], dict[tuple[str, str], str], str]:
    parent_run_id = mlflow.get_last_run_id(parent=parent_run_id)
    previous_runs = mlflow.search_runs()
    completed_runs = list(previous_runs[previous_runs["status"] == "FINISHED"]["name"])
    completed_runs = [tuple(run.split("_")) for run in completed_runs]
    incomplete_runs = {
        tuple(run["name"].split(_)): run["run_id"]
        for run in previous_runs[previous_runs["status"] == "RUNNING"].iterrows()
    }
    return completed_runs, incomplete_runs, parent_run_id
