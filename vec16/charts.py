import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK = "#1f2937"
ACCENT = "#2563eb"
WARM = "#ea580c"
MUTED = "#9ca3af"
PALETTE = ["#2563eb", "#ea580c", "#16a34a", "#9333ea", "#db2777", "#0891b2", "#ca8a04"]


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, color=INK, fontsize=12, loc="left")
    ax.set_xlabel(xlabel, color=INK)
    ax.set_ylabel(ylabel, color=INK)
    ax.grid(alpha=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def block(res, key):
    return res.get(key) or res["mean"]


def points(res, kind, key="mean"):
    return sorted(((int(d), v) for d, v in block(res, key).get(kind, {}).items()), key=lambda x: x[0])


def bytes_of(dim, kind):
    return dim * 4 if kind == "float" else dim / 8


def curve(path, students, baselines, bits=128, title="Retrieval quality against bytes stored per document", key="mean", teacher=None, late=None):
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    for i, (label, res) in enumerate(students):
        c = PALETTE[i % len(PALETTE)]
        fp = points(res, "float", key)
        bp = points(res, "binary", key)
        ax.plot([bytes_of(d, "float") for d, _ in fp], [v for _, v in fp], "-o", color=c, ms=4, label=f"{label} float")
        ax.plot([bytes_of(d, "binary") for d, _ in bp], [v for _, v in bp], "--s", color=c, ms=4, label=f"{label} binary")
    for j, (label, res, mrl) in enumerate(baselines):
        full = res["dim"]
        b = block(res, key)
        if mrl:
            fp = points(res, "float", key)
            ax.plot([bytes_of(d, "float") for d, _ in fp], [v for _, v in fp], ":", color=MUTED, lw=1)
            bp = points(res, "binary", key)
            ax.plot([bytes_of(d, "binary") for d, _ in bp], [v for _, v in bp], ":", color=MUTED, lw=1)
        if str(full) in b.get("float", {}):
            ax.scatter([bytes_of(full, "float")], [b["float"][str(full)]], marker="^", color=MUTED, edgecolor=INK, zorder=3, label="public embedders, float32" if j == 0 else None)
            ax.annotate(label, (bytes_of(full, "float"), b["float"][str(full)]), fontsize=7, color=INK, xytext=(4, 2), textcoords="offset points")
        if str(full) in b.get("binary", {}):
            ax.scatter([bytes_of(full, "binary")], [b["binary"][str(full)]], marker="v", color=MUTED, edgecolor=INK, zorder=3, label="public embedders, binary" if j == 0 else None)
    if teacher:
        tl, tr = teacher
        fp = points(tr, "float", key)
        ax.plot([bytes_of(d, "float") for d, _ in fp], [v for _, v in fp], "-", color=INK, lw=1, alpha=0.6, label=f"{tl} float")
        bp = points(tr, "binary", key)
        ax.plot([bytes_of(d, "binary") for d, _ in bp], [v for _, v in bp], "--", color=INK, lw=1, alpha=0.6, label=f"{tl} binary")
    if late:
        ll, score, nbytes = late
        if score is not None:
            ax.scatter([nbytes], [score], marker="*", s=120, color=WARM, edgecolor=INK, zorder=4, label=f"{ll}, late interaction")
    ax.axvline(bits / 8, color=WARM, lw=1, alpha=0.7)
    ax.text(bits / 8, ax.get_ylim()[0], f" {bits // 8} bytes", color=WARM, fontsize=8, va="bottom")
    ax.set_xscale("log", base=2)
    _style(ax, title, "bytes per document (log scale)", "NanoBEIR nDCG@10")
    ax.legend(fontsize=7, ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def mrl(path, label, res, base=None, key="mean"):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    names = {"float": "float32", "binary": "binary (Hamming)", "asym": "float query, binary docs", "rescore": "binary, top 100 rescored"}
    for kind, c, m in (("float", ACCENT, "-o"), ("binary", WARM, "--s"), ("asym", "#16a34a", ":^"), ("rescore", "#9333ea", "-.d")):
        p = points(res, kind, key)
        if p:
            ax.plot([d for d, _ in p], [v for _, v in p], m, color=c, ms=4, label=names[kind])
    if base:
        full = base["full_dim"]
        v = block(base, key)["float"].get(str(full))
        if v is not None:
            ax.axhline(v, color=MUTED, ls="-.", lw=1, label="starting LLM, mean pooled")
    ax.set_xscale("log", base=2)
    _style(ax, f"{label}: quality by vector size", "dimensions", "NanoBEIR nDCG@10")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def scaling(path, title, xlabel, series):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for i, (name, xs, ys) in enumerate(series):
        ax.plot(xs, ys, "-o", color=PALETTE[i % len(PALETTE)], ms=5, label=name)
        for x, y in zip(xs, ys):
            ax.annotate(f"{y * 100:.1f}", (x, y), fontsize=7, color=INK, xytext=(0, 5), textcoords="offset points", ha="center")
    ax.set_xscale("log")
    _style(ax, title, xlabel, "NanoBEIR nDCG@10")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def bench(path, rows):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    labels = [r["label"] for r in rows]
    mem = [r["bytes"] / 1024**2 for r in rows]
    qps = [r["qps"] for r in rows]
    for ax, vals, t, yl in ((axes[0], mem, "Index size", "MB"), (axes[1], qps, "Queries per second, CPU", "queries per second")):
        ax.bar(range(len(rows)), vals, color=[PALETTE[i % len(PALETTE)] for i in range(len(rows))])
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels(labels, fontsize=7, rotation=20, ha="right")
        ax.set_yscale("log")
        for x, v in enumerate(vals):
            ax.text(x, v, f"{v:,.0f}" if v >= 10 else f"{v:.2f}", ha="center", va="bottom", fontsize=7, color=INK)
        _style(ax, t, "", yl)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def bars(path, title, groups, series, ylabel="NanoBEIR nDCG@10"):
    fig, ax = plt.subplots(figsize=(max(6, 1.4 * len(groups) + 2), 4.5))
    w = 0.8 / max(1, len(series))
    for i, (name, vals) in enumerate(series):
        xs = [g + i * w for g in range(len(groups))]
        ax.bar(xs, vals, width=w, color=PALETTE[i % len(PALETTE)], label=name)
        for x, v in zip(xs, vals):
            ax.text(x, v, f"{v * 100:.1f}", ha="center", va="bottom", fontsize=7, color=INK)
    ax.set_xticks([g + w * (len(series) - 1) / 2 for g in range(len(groups))])
    ax.set_xticklabels(groups, fontsize=8)
    _style(ax, title, "", ylabel)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def loss_curve(path, label, hist):
    tr = [h for h in hist if "loss" in h]
    dv = [h for h in hist if "dev_mixed" in h]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot([h["step"] for h in tr], [h["loss"] for h in tr], color=ACCENT, lw=1, label="train loss")
    _style(ax, f"{label}: training", "step", "loss")
    if dv:
        ax2 = ax.twinx()
        ax2.plot([h["step"] for h in dv], [h["dev_mixed"] for h in dv], "o-", color=WARM, ms=4, label="dev nDCG@10 (float and binary mean)")
        ax2.set_ylabel("dev nDCG@10", color=WARM)
        ax2.spines["top"].set_visible(False)
        ax2.legend(fontsize=8, frameon=False, loc="lower right")
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
