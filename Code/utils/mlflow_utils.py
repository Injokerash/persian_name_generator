from mlflow.utils.logging_utils import eprint
from mlflow.entities import RunStatus
from mlflow.tracking import MlflowClient
from mlflow.tracking.fluent import _get_experiment_id
from mlflow.utils import mlflow_tags
import git

def already_ran(entry_point_name, parameters):
    repo = git.Repo(search_parent_directories=True)
    git_commit = repo.head.object.hexsha

    experiment_id = _get_experiment_id()

    client = MlflowClient()
    all_runs = reversed(client.search_runs([experiment_id]))
    for run in all_runs:
        tags = run.data.tags
        if tags.get(mlflow_tags.MLFLOW_PROJECT_ENTRY_POINT, None) != entry_point_name:
            continue

        match_failed = False
        for param_key, param_value in parameters.items():
            run_value = run.data.params.get(param_key)
            if str(run_value) != str(param_value):
                match_failed = True
                break
        if match_failed:
            continue
        if run.info.to_proto().status != RunStatus.FINISHED:
            eprint(
                ("Run matched, but is not FINISHED, so skipping (run_id={}, status={})").format(
                    run.info.run_id, run.info.status
                )
            )
            continue

        previous_version = tags.get(mlflow_tags.MLFLOW_GIT_COMMIT, None)
        if git_commit != previous_version:
            eprint(
                "Run matched, but has a different source version, so skipping "
                f"(found={previous_version}, expected={git_commit})"
            )
            continue
        return client.get_run(run.info.run_id)
    eprint("No matching run has been found.")
    return None