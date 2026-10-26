import json
import os
import random
import time
from pathlib import Path

import yaml

PKG = Path(__file__).resolve().parent
REPO = PKG.parent
ENV = "VEC16_"


def env(name, default=None):
    return os.environ.get(ENV + name, default)


def log(*parts):
    print(time.strftime("%H:%M:%S"), *parts, flush=True)


def detect_platform():
    if os.environ.get("RUNPOD_POD_ID"):
        return "runpod"
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        return "kaggle"
    if os.environ.get("COLAB_RELEASE_TAG") or Path("/content/sample_data").is_dir():
        return "colab"
    return "local"


def detect_profile(platform):
    return env("PROFILE") or ("full" if platform == "runpod" else "small")


def detect_work(platform, profile):
    if env("WORK"):
        base = Path(env("WORK"))
    elif platform == "runpod":
        base = Path("/workspace/vec16_work")
    elif platform == "colab":
        drive = Path("/content/drive/MyDrive")
        base = drive / "vec16_work" if drive.is_dir() else Path("/content/vec16_work")
    elif platform == "kaggle":
        base = Path("/kaggle/working/vec16_work")
    else:
        base = REPO / "work"
    return base / profile


def config_path():
    return Path(env("CONFIG")) if env("CONFIG") else PKG / "config.yaml"


def load_cfg():
    cfg = yaml.safe_load(config_path().read_text())
    platform = detect_platform()
    profile = detect_profile(platform)
    if profile not in cfg["profiles"]:
        raise SystemExit(f"unknown profile {profile}")
    run = dict(cfg["profiles"][profile])
    if env("MODELS"):
        keys = [k.strip() for k in env("MODELS").split(",") if k.strip()]
        bad = [k for k in keys if k not in cfg["models"]]
        if bad:
            raise SystemExit(f"unknown models in {ENV}MODELS: {bad}; choose from {list(cfg['models'])}")
        run["models"] = keys
        run["ablation_models"] = [k for k in (run.get("ablation_models") or []) if k in keys]
    if env("BASELINES"):
        run["baselines"] = [k.strip() for k in env("BASELINES").split(",") if k.strip()]
    work = detect_work(platform, profile)
    paths = {"work": work}
    for name in ("data", "teacher", "runs", "results", "assets", "publish", "cache"):
        paths[name] = work / name
        paths[name].mkdir(parents=True, exist_ok=True)
    cfg.update(platform=platform, profile=profile, run=run, paths=paths)
    return cfg


def publish_enabled():
    return env("PUBLISH", "1") == "1"


def hf_token():
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    try:
        from huggingface_hub import get_token

        return get_token()
    except Exception:
        return None


def hf_login():
    token = hf_token()
    if not token:
        return None
    os.environ["HF_TOKEN"] = token
    try:
        from huggingface_hub import login

        login(token=token, add_to_git_credential=False)
    except Exception:
        pass
    return token


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
    tmp.replace(path)


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    return rows


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


def gpu_info():
    import torch

    if not torch.cuda.is_available():
        return {"cuda": False, "bf16": False, "cc_major": 0}
    p = torch.cuda.get_device_properties(0)
    return {
        "cuda": True,
        "name": p.name,
        "vram_gb": round(p.total_memory / 1024**3, 1),
        "cc": f"{p.major}.{p.minor}",
        "cc_major": p.major,
        "bf16": p.major >= 8,
        "count": torch.cuda.device_count(),
    }


def device():
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def train_precision(mcfg):
    info = gpu_info()
    if info["bf16"]:
        return "bf16"
    if not info["cuda"] or mcfg.get("no_fp16"):
        return "fp32"
    return "fp16"


def infer_dtype(mcfg=None):
    import torch

    info = gpu_info()
    if info["bf16"]:
        return torch.bfloat16
    if info["cuda"] and not (mcfg or {}).get("no_fp16"):
        return torch.float16
    return torch.float32


def eval_dtype(mcfg=None):
    import torch

    return torch.bfloat16 if gpu_info()["bf16"] else torch.float32


def seed_all(seed):
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def release():
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def mrl_dims(cfg, full):
    return sorted({d for d in cfg["train"]["matryoshka_dims"] if d < full} | {full}, reverse=True)


def bin_dims(cfg, full):
    return sorted({d for d in cfg["train"]["binary_dims"] if d <= full}, reverse=True)


def headline_bits(cfg):
    return int(cfg["verdict_bits"])


def run_dir(cfg, mkey, vname, seed):
    return cfg["paths"]["runs"] / mkey / f"{vname}__s{seed}"


def model_dir(cfg, mkey):
    return cfg["paths"]["runs"] / mkey


def repo_name(cfg, mkey):
    return f"{cfg['project']['name']}_{cfg['models'][mkey]['tail']}{cfg['run']['repo_suffix']}"


def dataset_name(cfg):
    return f"{cfg['project']['name']}_{cfg['project']['dataset_tail']}{cfg['run']['repo_suffix']}"


def variant_spec(cfg, vname, headline_teacher=None):
    base = dict(cfg["defaults"])
    if vname in cfg["variants"]:
        base.update(cfg["variants"][vname])
    elif vname in cfg["ablations"]:
        base["teacher"] = headline_teacher or "fused"
        base.update(cfg["ablations"][vname])
    elif vname.startswith("scale_"):
        base["teacher"] = headline_teacher or "fused"
        base["train_queries"] = int(vname.split("_", 1)[1])
    else:
        raise KeyError(vname)
    return base


def train_queries(cfg, mkey, spec):
    n = spec.get("train_queries") or cfg["models"][mkey].get("train_queries") or cfg["run"].get("train_queries")
    return int(n) if n else None


def scale_variants(cfg, mkey):
    if mkey not in (cfg["run"].get("scaling_models") or []):
        return []
    return [f"scale_{n}" for n in cfg["run"].get("scaling_queries") or []]


def status_path(cfg):
    return cfg["paths"]["work"] / "status.json"


def update_status(cfg, key, value):
    p = status_path(cfg)
    st = read_json(p) if p.exists() else {}
    st[key] = value
    st["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    write_json(p, st)


def model_status(cfg, mkey, **fields):
    p = status_path(cfg)
    st = read_json(p) if p.exists() else {}
    models = st.setdefault("models", {})
    cur = models.setdefault(mkey, {})
    cur.update(fields)
    st["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    write_json(p, st)


def linear_schedule(warm, total):
    def f(s):
        if s < warm:
            return (s + 1) / warm
        return max(0.0, (total - s) / max(1, total - warm))

    return f


def model_kwargs(dtype=None, attn=None):
    kw = {}
    if dtype is not None:
        kw["dtype"] = dtype
    if attn:
        kw["attn_implementation"] = attn
    return kw


if __name__ == "__main__":
    print(detect_platform())
