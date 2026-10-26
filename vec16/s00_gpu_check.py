import importlib
import sys
import time

import torch

from vec16.common import env, gpu_info, hf_login, load_cfg, log, publish_enabled, update_status, write_json

PACKAGES = ("transformers", "sentence_transformers", "datasets", "huggingface_hub", "accelerate", "pylate", "onnx", "onnxruntime")


def versions():
    out = {"python": sys.version.split()[0], "torch": torch.__version__, "cuda_runtime": torch.version.cuda}
    for pkg in PACKAGES:
        try:
            out[pkg] = importlib.import_module(pkg).__version__
        except Exception as ex:
            out[pkg] = f"missing: {ex}"
    return out


def gpu_smoke(info):
    dtype = torch.bfloat16 if info["bf16"] else torch.float16
    x = torch.randn(2048, 2048, device="cuda", dtype=dtype, requires_grad=True)
    w = torch.randn(2048, 2048, device="cuda", dtype=dtype, requires_grad=True)
    torch.cuda.synchronize()
    t0 = time.time()
    y = (x @ w).float().pow(2).mean()
    y.backward()
    torch.cuda.synchronize()
    ok = bool(torch.isfinite(x.grad).all().item() and torch.isfinite(w.grad).all().item())
    free, _ = torch.cuda.mem_get_info()
    return {"matmul_ok": ok, "matmul_ms": round((time.time() - t0) * 1000, 2), "free_gb": round(free / 1024**3, 1)}


def gated_ids(cfg):
    run = cfg["run"]
    ids = [cfg["models"][m]["id"] for m in run["models"] if cfg["models"][m].get("gated")]
    ids += [cfg["baselines"][b]["id"] for b in run["baselines"] if cfg["baselines"][b].get("gated")]
    return ids


def hf_checks(cfg):
    out = {}
    token = hf_login()
    if not token:
        out["hf_token"] = False
        return out
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    try:
        out["hf_user"] = api.whoami()["name"]
        out["hf_token"] = True
    except Exception as ex:
        out["hf_token"] = False
        out["hf_error"] = repr(ex)
        return out
    gated = {}
    for rid in gated_ids(cfg):
        try:
            api.auth_check(rid)
            gated[rid] = "ok"
        except Exception as ex:
            gated[rid] = f"no access: {type(ex).__name__}"
    out["gated"] = gated
    return out


def main():
    cfg = load_cfg()
    info = gpu_info()
    info.update(platform=cfg["platform"], profile=cfg["profile"], work=str(cfg["paths"]["work"]), **versions())
    if not info["cuda"]:
        log("no CUDA GPU found")
        if env("ALLOW_CPU") != "1":
            sys.exit(1)
    else:
        info.update(gpu_smoke(info))
        if not info["matmul_ok"]:
            log("GPU matmul produced non finite values")
            sys.exit(1)
    info.update(hf_checks(cfg))
    write_json(cfg["paths"]["work"] / "env.json", info)
    for k, v in info.items():
        log(f"{k}: {v}")
    missing = [p for p in PACKAGES if str(info.get(p, "")).startswith("missing")]
    if missing:
        log("missing packages:", ", ".join(missing), "run bash setup.sh")
        sys.exit(1)
    bad = [k for k, v in (info.get("gated") or {}).items() if v != "ok"]
    if bad:
        log("gated models without access:", ", ".join(bad))
        log("accept the license on huggingface.co or drop them with VEC16_MODELS / VEC16_BASELINES")
        sys.exit(1)
    if not info.get("hf_token"):
        log("no HuggingFace login found; set HF_TOKEN")
        if publish_enabled():
            sys.exit(1)
    log(f"students this run: {', '.join(cfg['run']['models'])}")
    log(f"baselines this run: {', '.join(cfg['run']['baselines'])}")
    update_status(cfg, "env", {"gpu": info.get("name"), "platform": cfg["platform"], "profile": cfg["profile"]})
    log("gpu check passed")


if __name__ == "__main__":
    main()
