from __future__ import annotations
import json, math, re
from pathlib import Path
from typing import Dict

_PATTERNS={
 "FID":re.compile(r"FID(?:/CIFAR100)?[:=]\s*([0-9.eE+-]+)"),
 "IS":re.compile(r"IS[:=]\s*([0-9.eE+-]+)"),
 "Recall":re.compile(r"(?:RECALL|Recall)[:=]\s*([0-9.eE+-]+)"),
 "Precision":re.compile(r"(?:PRECISION|Precision)[:=]\s*([0-9.eE+-]+)"),
 "KID":re.compile(r"KID[:=]\s*([0-9.eE+-]+)"),
}

def _put(out,key,value):
    try:
        v=float(value)
        if math.isfinite(v): out[key]=v
    except Exception: pass

def parse_text_metrics(path: Path)->Dict[str,float]:
    if not path.exists(): return {}
    text=path.read_text(encoding="utf-8",errors="replace"); out={}
    for k,p in _PATTERNS.items():
        m=p.findall(text)
        if m: _put(out,k,m[-1])
    return out

def _flatten(prefix,obj,out):
    if isinstance(obj,dict):
        if "mean" in obj and isinstance(obj["mean"],(int,float)):
            _put(out,prefix,obj["mean"]); return
        for k,v in obj.items(): _flatten(f"{prefix}/{k}" if prefix else str(k),v,out)
    elif isinstance(obj,(int,float)): _put(out,prefix,obj)

# Sample counts, seeds and protocol constants are provenance, not results.
# They belong in the run's config and the report's JSON, not plotted next to
# FID as if they were something a reader should compare across methods.
_BOOKKEEPING = ("num_generated", "num_reference", "generated", "reference",
                "seed", "samples", "label_histogram")


def _is_bookkeeping(key: str) -> bool:
    return key.rsplit("/", 1)[-1] in _BOOKKEEPING

def collect_metrics(run_dir: Path)->Dict[str,float]:
    out={}
    semantic=run_dir/"semantic_metrics.json"
    if semantic.exists():
        p=json.loads(semantic.read_text())
        for k in ("js","rare_mode_mass","coarse_consistency","memorization","num_generated"):
            if k in p: _put(out,f"semantic/{k}",p[k])
        _flatten("generation",p.get("generation",{}),out); _flatten("safety",p.get("safety",{}),out)
    text_candidates = [run_dir/"stdout.log", *run_dir.glob("res_ema_*.txt"), *run_dir.glob("eval.txt")]
    for path in text_candidates:
        for k,v in parse_text_metrics(path).items(): out[f"generation/{k}"]=v
    for path in run_dir.rglob("metrics*.json"):
        if path.name=="metrics.collected.json": continue
        try: payload = json.loads(path.read_text())
        except Exception: continue
        if not isinstance(payload, dict): continue
        if isinstance(payload.get("metrics"), dict):
            # The shared CCUA evaluator wraps FID/IS/KID/PRD in a "metrics" key
            # alongside "protocol"/"label_histogram" metadata. Flattening the
            # whole payload buried every metric two levels deep as
            # generation/metrics/FID and swept protocol constants in as if
            # they were metrics; unwrap to the clean generation/FID instead.
            _flatten("generation", payload["metrics"], out)
        elif isinstance(payload.get("groups"), dict):
            # metrics.per_class.json: keep the Many/Medium/Few tail summary,
            # skip the 100-class per_class breakdown and protocol metadata so
            # they don't flood the W&B summary with noise.
            _flatten("generation/tail", payload["groups"], out)
        else:
            _flatten("generation", payload, out)
    return {k: v for k, v in out.items() if not _is_bookkeeping(k)}

def latest_tensorboard_scalars(logdir: Path):
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except Exception: return {}
    if not list(logdir.rglob("events.out.tfevents.*")): return {}
    try:
        acc=EventAccumulator(str(logdir),size_guidance={"scalars":1}); acc.Reload(); out={}
        for tag in acc.Tags().get("scalars",[]):
            ev=acc.Scalars(tag)
            if ev: out[tag]=(int(ev[-1].step),float(ev[-1].value))
        return out
    except Exception: return {}
