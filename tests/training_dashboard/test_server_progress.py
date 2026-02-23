from __future__ import annotations

from training_dashboard.server import TrainingJob


def test_progress_mapping_promotes_nodes_by_step_and_epoch() -> None:
    job = TrainingJob()
    job.start(cmd=["python3", "-m", "pipelines.run_local_training_pipeline"], run_id="unit-run")

    job.append_log("[1/6] train_patchtst")
    first = job.snapshot()
    assert first["active_step"] == "train_patchtst"
    assert first["step_index"] == 1
    assert first["step_total"] == 6
    assert first["live_nodes"]["data-prep"]["status"] == "running"
    assert first["live_nodes"]["dqvl"]["status"] == "running"
    assert first["live_nodes"]["patchtst"]["status"] == "idle"

    job.append_log("[PatchTST][Epoch 1] train=0.123 val=0.456")
    promoted = job.snapshot()
    assert promoted["live_nodes"]["data-prep"]["status"] == "done"
    assert promoted["live_nodes"]["dqvl"]["status"] == "done"
    assert promoted["live_nodes"]["patchtst"]["status"] == "running"

    job.append_log("[2/6] train_swinmae")
    second = job.snapshot()
    assert second["live_nodes"]["patchtst"]["status"] == "done"
    assert second["live_nodes"]["swinmae"]["status"] == "running"


def test_failure_marks_active_node_and_orchestrator_fail() -> None:
    job = TrainingJob()
    job.start(cmd=["python3", "-m", "pipelines.run_local_training_pipeline"], run_id="unit-run")
    job.append_log("[2/6] train_swinmae")
    job.finish(return_code=1)

    result = job.snapshot()
    assert result["state"] == "failed"
    assert result["live_nodes"]["swinmae"]["status"] == "fail"
    assert result["live_nodes"]["orchestrator"]["status"] == "fail"
