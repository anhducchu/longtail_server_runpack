"""Load YAML files and expand them into concrete experiment tasks."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

from .utils import deep_merge, expand_env, stable_id


@dataclass(frozen=True)
class Task:
    id: str
    campaign: str
    stage: str
    adapter: str
    method: str
    seed: int
    priority: int
    dataset: dict[str, Any]
    train: dict[str, Any]
    eval: dict[str, Any]
    method_config: dict[str, Any]
    repository: dict[str, Any]
    runtime: dict[str, Any]
    retry: dict[str, Any]
    tags: list[str] = field(default_factory=list)
    run_dir: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LoadedCampaign:
    root: Path
    raw: dict[str, Any]
    server: dict[str, Any]
    tasks: list[Task]


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return expand_env(yaml.safe_load(f) or {})


def _resolve(root: Path, value: str) -> str:
    if not value:
        return value
    p = Path(value).expanduser()
    return str(p if p.is_absolute() else (root / p).resolve())


def _resolve_runtime(root: Path, server: dict[str, Any]) -> dict[str, Any]:
    runtime = server.get("runtime", {})
    runtime = dict(runtime)
    for key in ("repos_root", "runs_root", "data_root"):
        runtime[key] = _resolve(root, str(runtime.get(key, f"./{key.replace('_root', '')}")))
    return runtime


def _resolve_setting_path(settings: dict[str, Any], key: str, root: Path) -> dict[str, Any]:
    resolved = dict(settings)
    if resolved.get(key):
        resolved[key] = _resolve(root, str(resolved[key]))
    return resolved


def _expand_stage(
    *,
    root: Path,
    stage: dict[str, Any],
    campaign_cfg: dict[str, Any],
    campaign_name: str,
    shared: dict[str, Any],
    repositories: dict[str, Any],
    runtime: dict[str, Any],
    retry: dict[str, Any],
) -> list[Task]:
    if not stage.get("enabled", True):
        return []

    adapter = stage.get("adapter", shared.get("adapter", "ccua"))
    uses_shared_backbone = adapter == shared.get("adapter")
    shared_train = shared.get("train", {}) if uses_shared_backbone else {}
    shared_eval = shared.get("eval", {}) if uses_shared_backbone else {}
    train = _resolve_setting_path(
        deep_merge(shared_train, stage.get("train", {})),
        "resume_checkpoint",
        root,
    )
    evaluate = deep_merge(shared_eval, stage.get("eval", {}))
    dataset = _resolve_setting_path(stage.get("dataset", {}), "root", root)
    seeds = stage.get("seeds", campaign_cfg.get("paired_seeds", [0, 1, 2]))

    tasks = []
    for raw_method in stage.get("methods", []):
        method_config = _resolve_setting_path(raw_method, "resume_checkpoint", root)
        method = method_config["name"]
        for seed in seeds:
            task_id = stable_id(campaign_name, stage["name"], method, seed)
            run_dir = (
                Path(runtime["runs_root"])
                / campaign_name
                / stage["name"]
                / method
                / f"seed_{seed}"
            )
            tasks.append(Task(
                id=task_id,
                campaign=campaign_name,
                stage=stage["name"],
                adapter=adapter,
                method=method,
                seed=int(seed),
                priority=int(
                    stage.get("priority", campaign_cfg.get("default_priority", 50))
                ),
                dataset=dataset,
                train=train,
                eval=evaluate,
                method_config=method_config,
                repository=repositories.get(adapter, {}),
                runtime=runtime,
                retry=retry,
                tags=list(stage.get("tags", [])),
                run_dir=str(run_dir),
            ))
    return tasks


def load_campaign(config_path: str | Path) -> LoadedCampaign:
    """Resolve one campaign file into the ordered tasks the scheduler runs."""
    config_path = Path(config_path).expanduser().resolve()
    root = config_path.parent.parent
    raw = load_yaml(config_path)
    campaign_cfg = raw.get("campaign", {})
    server_path = _resolve(root, campaign_cfg.get("server_config", "configs/server.yaml"))
    server = load_yaml(Path(server_path))
    runtime = _resolve_runtime(root, server)

    server["runtime"] = runtime
    retry = server.get("retry", {})
    campaign_name = campaign_cfg.get("name", config_path.stem)
    shared = raw.get("shared_backbone", {})
    tasks: list[Task] = []
    for stage in raw.get("stages", []):
        tasks.extend(_expand_stage(
            root=root,
            stage=stage,
            campaign_cfg=campaign_cfg,
            campaign_name=campaign_name,
            shared=shared,
            repositories=raw.get("repositories", {}),
            runtime=runtime,
            retry=retry,
        ))

    tasks.sort(key=lambda t: (-t.priority, t.seed, t.stage, t.method))
    return LoadedCampaign(root=root, raw=raw, server=server, tasks=tasks)


def _select_resume_tasks(
    campaign: LoadedCampaign,
    method: str,
    seed: int | None,
    stage: str | None,
) -> list[Task]:
    tasks = [
        task
        for task in campaign.tasks
        if task.method == method
        and (seed is None or task.seed == seed)
        and (stage is None or task.stage == stage)
    ]
    if not tasks:
        raise ValueError(
            f"no campaign task matches method={method!r}, "
            f"seed={seed!r}, stage={stage!r}"
        )
    if any(task.adapter != "ccua" for task in tasks):
        raise ValueError("external resume supports CCUA-DDPM tasks only")
    return tasks


def _validate_resume_template(template: str, tasks: list[Task], seed: int | None) -> None:
    if seed is None and len(tasks) > 1 and "{seed}" not in template:
        raise ValueError(
            "multiple seeds match; use --resume-seed or include {seed} in the checkpoint path"
        )
    if len({task.stage for task in tasks}) > 1 and "{stage}" not in template:
        raise ValueError("multiple stages match; include {stage} in the checkpoint path")
    if len({str(task.dataset.get('name', '')) for task in tasks}) > 1 and "{dataset}" not in template:
        raise ValueError("multiple datasets match; include {dataset} in the checkpoint path")


def _checkpoint_path(template: str, task: Task) -> str:
    try:
        value = template.format(
            seed=task.seed,
            method=task.method,
            stage=task.stage,
            dataset=task.dataset.get("name", ""),
        )
    except (KeyError, ValueError) as exc:
        raise ValueError(
            "checkpoint paths may use only {seed}, {method}, {stage}, and {dataset}"
        ) from exc
    return str(Path(value).expanduser().resolve())


def apply_run_overrides(
    campaign: LoadedCampaign,
    *,
    gpus: str | None = None,
    per_gpu: int | str | None = None,
    jobs: int | str | None = None,
    resume_checkpoint: str | None = None,
    resume_method: str | None = None,
    resume_seed: int | None = None,
    resume_stage: str | None = None,
    resume_step: int | None = None,
) -> None:
    """Apply command-line machine limits and explicit resume settings."""
    machine = campaign.server.setdefault("machine", {})
    if gpus:
        machine["gpu_ids"] = [int(value) for value in gpus.split(",") if value.strip()]
    if per_gpu is not None:
        machine["tasks_per_gpu"] = per_gpu
    if jobs is not None:
        machine["max_concurrent"] = jobs

    selectors_used = any(
        value is not None
        for value in (resume_method, resume_seed, resume_stage, resume_step)
    )
    if not resume_checkpoint:
        if selectors_used:
            raise ValueError("resume options require --resume-checkpoint")
        return
    if not resume_method:
        raise ValueError("--resume-checkpoint requires --resume-method")

    selected = _select_resume_tasks(campaign, resume_method, resume_seed, resume_stage)
    _validate_resume_template(resume_checkpoint, selected, resume_seed)
    selected_ids = {task.id for task in selected}
    updated = []
    for task in campaign.tasks:
        if task.id not in selected_ids:
            updated.append(task)
            continue
        method_config = dict(task.method_config)
        method_config["resume_checkpoint"] = _checkpoint_path(resume_checkpoint, task)
        if resume_step is not None:
            method_config["resume_step"] = resume_step
        updated.append(replace(task, method_config=method_config))
    campaign.tasks = updated
