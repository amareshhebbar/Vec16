import os
import platform
import statistics
import time

import numpy as np

from vec16 import charts
from vec16.common import load_cfg, log, update_status, write_json

FLOAT_DIM = 1024


def timed_search(index, q, k=10):
    t0 = time.time()
    index.search(q, k)
    return time.time() - t0


def latency(index, q, k=10, n=50):
    import faiss

    faiss.omp_set_num_threads(1)
    ts = []
    for i in range(min(n, len(q))):
        t0 = time.time()
        index.search(q[i : i + 1], k)
        ts.append((time.time() - t0) * 1000)
    faiss.omp_set_num_threads(os.cpu_count() or 1)
    return statistics.median(ts)


def float_index(n, dim, rng, kind):
    import faiss

    x = rng.standard_normal((n, dim), dtype=np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    if kind == "float32":
        idx = faiss.IndexFlatIP(dim)
    else:
        idx = faiss.IndexScalarQuantizer(dim, faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_INNER_PRODUCT)
        idx.train(x[: min(n, 100000)])
    t0 = time.time()
    for i in range(0, n, 500000):
        idx.add(x[i : i + 500000])
    return idx, time.time() - t0


def binary_index(n, bits, rng):
    import faiss

    idx = faiss.IndexBinaryFlat(bits)
    t0 = time.time()
    for i in range(0, n, 2000000):
        idx.add(rng.integers(0, 256, size=(min(2000000, n - i), bits // 8), dtype=np.uint8))
    return idx, time.time() - t0


def measure(label, kind, n, dim_or_bits, idx, build, q):
    total = timed_search(idx, q)
    row = {
        "label": label,
        "kind": kind,
        "n": n,
        "dims": dim_or_bits,
        "bytes": int(idx.ntotal * (dim_or_bits * 4 if kind == "float32" else dim_or_bits if kind == "int8" else dim_or_bits / 8)),
        "build_seconds": round(build, 2),
        "qps": round(len(q) / total, 1),
        "latency_ms": round(latency(idx, q), 2),
        "measured": True,
    }
    log(f"bench {label} n={n:,} size {row['bytes'] / 1024**2:,.1f} MB qps {row['qps']:,} p50 {row['latency_ms']} ms")
    return row


def main():
    import faiss

    cfg = load_cfg()
    run = cfg["run"]
    out = cfg["paths"]["results"] / "bench.json"
    if out.exists():
        log("benchmark already done")
        return
    rng = np.random.default_rng(0)
    nq = run.get("bench_queries", 1000)
    faiss.omp_set_num_threads(os.cpu_count() or 1)
    rows = []
    for n in run.get("bench_sizes") or [100000]:
        fq = rng.standard_normal((nq, FLOAT_DIM), dtype=np.float32)
        fq /= np.linalg.norm(fq, axis=1, keepdims=True)
        for kind in ("float32", "int8"):
            label = f"{kind} {FLOAT_DIM} dims"
            if n <= run.get("bench_float_max", 1000000):
                idx, build = float_index(n, FLOAT_DIM, rng, kind)
                rows.append(measure(label, kind, n, FLOAT_DIM, idx, build, fq))
                del idx
            else:
                per = FLOAT_DIM * 4 if kind == "float32" else FLOAT_DIM
                rows.append({"label": label, "kind": kind, "n": n, "dims": FLOAT_DIM, "bytes": n * per, "qps": None, "latency_ms": None, "measured": False})
                log(f"bench {label} n={n:,} size {n * per / 1024**3:,.1f} GB computed, not loaded")
        for bits in (1024, 256, cfg["verdict_bits"]):
            idx, build = binary_index(n, bits, rng)
            bq = rng.integers(0, 256, size=(nq, bits // 8), dtype=np.uint8)
            rows.append(measure(f"binary {bits} bits", "binary", n, bits, idx, build, bq))
            del idx
    info = {
        "cpu": platform.processor() or platform.machine(),
        "threads": os.cpu_count(),
        "faiss": faiss.__version__,
        "queries": nq,
        "note": "Flat exact indexes on random vectors with the same shape as real embeddings. Flat search time does not depend on the values, so quality comes from NanoBEIR and speed and size from here.",
        "rows": rows,
    }
    write_json(out, info)
    big = max(r["n"] for r in rows)
    charts.bench(cfg["paths"]["assets"] / "bench.png", [dict(r, qps=r["qps"] or 0.0, label=f"{r['label']}{'' if r['measured'] else ' (size only)'}") for r in rows if r["n"] == big])
    update_status(cfg, "bench", {"sizes": run.get("bench_sizes"), "rows": len(rows)})


if __name__ == "__main__":
    main()
