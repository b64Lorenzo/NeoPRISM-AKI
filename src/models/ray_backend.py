"""Local Ray tasks around pure numerical classifiers, never around clinical files.

Task and resource APIs: https://docs.ray.io/en/latest/ray-core/tasks.html
Ray is imported only on an explicitly requested training execution path.
"""

import os
import tempfile
from pathlib import Path

from src.models.factory import create_model, fit_numeric, restore_model
from src.models.neural_network.nnUtils import features, integer, labels, real_number


def _fit_worker(model_type, x, y, constructor, fit_options):
    import ray

    payload = fit_numeric(model_type, x, y, constructor, fit_options)
    return {
        "model": payload,
        "execution": {
            "backend": "ray",
            "worker_pid": os.getpid(),
            "worker_id": str(ray.get_runtime_context().get_worker_id()),
            "ray_version": ray.__version__,
        },
    }


def fit_with_ray(
    model_type,
    x,
    y,
    constructor,
    fit_options,
    *,
    num_cpus=1,
    cpus_per_task=1.0,
    timeout_seconds=300.0,
    temp_directory=None,
):
    """Return portable model payload and provenance from a real local worker.

    This function owns its local runtime. It refuses a pre-initialized Ray
    context instead of silently sending data to, or shutting down, another job.
    There is no local fallback when Ray fails. Inference never uses this path.
    """
    num_cpus = integer(num_cpus, "num_cpus")
    cpus_per_task = real_number(cpus_per_task, "cpus_per_task", positive=True)
    timeout_seconds = real_number(timeout_seconds, "timeout_seconds", positive=True)
    if cpus_per_task > num_cpus:
        raise ValueError("cpus_per_task must not exceed num_cpus")
    create_model(model_type, constructor)
    x = features(x)
    y = labels(y, len(x))
    import ray

    if ray.is_initialized():
        raise RuntimeError(
            "Ray is already initialized; use a dedicated driver process for this local training job"
        )
    os.environ.setdefault("RAY_USAGE_STATS_ENABLED", "0")
    runtime_dir = (
        Path(temp_directory)
        if temp_directory is not None
        else Path(tempfile.gettempdir()) / "neoprism-ray"
    )
    # Ray's native Windows processes do not consistently accept long filenames.
    # Account for its session directory and the longest core-worker log name.
    longest_log = (
        runtime_dir.resolve()
        / "session_9999-99-99_99-99-99_999999_4294967295"
        / "logs"
        / ("python-core-worker-" + "f" * 64 + "_4294967295.log")
    )
    if os.name == "nt" and len(str(longest_log)) >= 260:
        raise ValueError(
            "Ray temp_directory is too long for native Windows; choose a shorter directory"
        )
    result = None
    try:
        ray.init(
            address="local",
            num_cpus=num_cpus,
            include_dashboard=False,
            log_to_driver=False,
            logging_level="ERROR",
            local_mode=False,
            _temp_dir=str(runtime_dir.resolve()),
        )
        reference = ray.remote(num_cpus=cpus_per_task, max_retries=0)(
            _fit_worker
        ).remote(
            model_type,
            x,
            y,
            dict(constructor),
            dict(fit_options),
        )
        try:
            result = ray.get(reference, timeout=timeout_seconds)
        except ray.exceptions.GetTimeoutError as error:
            ray.cancel(reference, force=True)
            raise RuntimeError("Ray training task exceeded timeout_seconds") from error
    except Exception as error:
        raise RuntimeError(f"Ray training failed: {error}") from error
    finally:
        ray.shutdown()
    if not isinstance(result, dict) or set(result) != {"model", "execution"}:
        raise ValueError("Malformed Ray worker result")
    restore_model(model_type, result["model"])
    execution = result["execution"]
    if (
        not isinstance(execution, dict)
        or execution.get("backend") != "ray"
        or type(execution.get("worker_pid")) is not int
        or execution["worker_pid"] <= 0
        or execution["worker_pid"] == os.getpid()
        or not execution.get("worker_id")
    ):
        raise ValueError("Worker provenance does not identify a separate Ray process")
    return result["model"], {
        **execution,
        "driver_pid": os.getpid(),
        "num_cpus": num_cpus,
        "cpus_per_task": cpus_per_task,
    }
