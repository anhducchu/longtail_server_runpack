from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List

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
    dataset: Dict[str, Any]
    train: Dict[str, Any]
    eval: Dict[str, Any]
    method_config: Dict[str, Any]
    repository: Dict[str, Any]
    runtime: Dict[str, Any]
    retry: Dict[str, Any]
    tags: List[str] = field(default_factory=list)
    run_dir: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LoadedCampaign:
    root: Path
    raw: Dict[str, Any]
    server: Dict[str, Any]
    tasks: List[Task]


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return expand_env(yaml.safe_load(f) or {})


def _resolve(root: Path, value: str) -> str:
    if not value:
        return value
    p = Path(value).expanduser()
    return str(p if p.is_absolute() else (root / p).resolve())


def load_campaign(config_path: str | Path) -> LoadedCampaign:
    config_path = Path(config_path).expanduser().resolve()
    root = config_path.parent.parent
    raw = load_yaml(config_path)
    campaign_cfg = raw.get("campaign", {})
    server_path = _resolve(root, campaign_cfg.get("server_config", "configs/server.yaml"))
    server = load_yaml(Path(server_path))

    runtime = server.get("runtime", {})
    runtime = dict(runtime)
    for key in ("repos_root", "runs_root", "data_root"):
        runtime[key] = _resolve(root, str(runtime.get(key, f"./{key.replace('_root', '')}")))

    server["runtime"] = runtime
    retry = server.get("retry", {})
    campaign_name = campaign_cfg.get("name", config_path.stem)
    repos = raw.get("repositories", {})
    shared = raw.get("shared_backbone", {})
    shared_train = shared.get("train", {})
    shared_eval = shared.get("eval", {})

    tasks: List[Task] = []
    for stage in raw.get("stages", []):
        if not stage.get("enabled", True):
            continue
        adapter = stage.get("adapter", shared.get("adapter", "ccua"))
        repo = repos.get(adapter, {})
        stage_train = deep_merge(shared_train if adapter == shared.get("adapter") else {}, stage.get("train", {}))
        stage_eval = deep_merge(shared_eval if adapter == shared.get("adapter") else {}, stage.get("eval", {}))
        if stage_train.get("resume_checkpoint"):
            stage_train["resume_checkpoint"] = _resolve(root, str(stage_train["resume_checkpoint"]))
        dataset = dict(stage.get("dataset", {}))
        if dataset.get("root"):
            dataset["root"] = _resolve(root, dataset["root"])
        seeds = stage.get("seeds", campaign_cfg.get("paired_seeds", [0, 1, 2]))
        for method_cfg in stage.get("methods", []):
            method_cfg = dict(method_cfg)
            if method_cfg.get("resume_checkpoint"):
                method_cfg["resume_checkpoint"] = _resolve(root, str(method_cfg["resume_checkpoint"]))
            method = method_cfg["name"]
            for seed in seeds:
                task_id = stable_id(campaign_name, stage["name"], method, seed)
                run_dir = str(Path(runtime["runs_root"]) / campaign_name / stage["name"] / method / f"seed_{seed}")
                tasks.append(Task(
                    id=task_id,
                    campaign=campaign_name,
                    stage=stage["name"],
                    adapter=adapter,
                    method=method,
                    seed=int(seed),
                    priority=int(stage.get("priority", campaign_cfg.get("default_priority", 50))),
                    dataset=dataset,
                    train=stage_train,
                    eval=stage_eval,
                    method_config=method_cfg,
                    repository=repo,
                    runtime=runtime,
                    retry=retry,
                    tags=list(stage.get("tags", [])),
                    run_dir=run_dir,
                ))

    tasks.sort(key=lambda t: (-t.priority, t.seed, t.stage, t.method))
    return LoadedCampaign(root=root, raw=raw, server=server, tasks=tasks)
