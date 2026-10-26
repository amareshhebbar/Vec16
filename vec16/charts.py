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


def points(res, kind):
    return sorted(((int(d), v) for d, v in res["mean"][kind].items()), key=lambda x: x[0])


def bytes_of(dim, kind):
    return dim * 4 if kind == "float" else dim / 8


def curve(path, students, baselines, bits=128, title="Retrieval quality against bytes stored per document"):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, (label, res) in enumerate(students):
        c = PALETTE[i % len(PALETTE)]
        fp = points(res, "float")
        bp = points(res, "binary")
        ax.plot([bytes_of(d, "float") for d, _ in fp], [v for _, v in fp], "-o", color=c, ms=4, label=f"{label} float")
        ax.plot([bytes_of(d, "binary") for d, _ in bp], [v for _, v in bp], "--s", color=c, ms=4, label=f"{label} binary")
    for j, (label, res, mrl) in enumerate(baselines):
        full = res["dim"]
        if mrl:
            fp = points(res, "float")
            ax.plot([bytes_of(d, "float") for d, _ in fp], [v for _, v in fp], ":", color=MUTED, lw=1)
        ax.scatter([bytes_of(full, "float")], [res["mean"]["float"][str(full)]], marker="^", color=MUTED, edgecolor=INK, zorder=3, label="public embedders, float32" if j == 0 else None)
        ax.annotate(label, (bytes_of(full, "float"), res["mean"]["float"][str(full)]), fontsize=7, color=INK, xytext=(4, 2), textcoords="offset points")
        if str(full) in res["mean"]["binary"]:
            ax.scatter([bytes_of(full, "binary")], [res["mean"]["binary"][str(full)]], marker="v", color=MUTED, edgecolor=INK, zorder=3, label="public embedders, binary" if j == 0 else None)
    ax.axvline(bits / 8, color=WARM, lw=1, alpha=0.7)
    ax.text(bits / 8, ax.get_ylim()[0], f" {bits // 8} bytes", color=WARM, fontsize=8, va="bottom")
    ax.set_xscale("log", base=2)
    _style(ax, title, "bytes per document (log scale)", "NanoBEIR nDCG@10")
    ax.legend(fontsize=7, ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def mrl(path, label, res, base=None):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for kind, c, m in (("float", ACCENT, "-o"), ("binary", WARM, "--s"), ("asym", "#16a34a", ":^")):
        p = points(res, kind)
        ax.plot([d for d, _ in p], [v for _, v in p], m, color=c, ms=4, label={"float": "float32", "binary": "binary (Hamming)", "asym": "float query, binary docs"}[kind])
    if base:
        full = base["full_dim"]
        ax.axhline(base["mean"]["float"][str(full)], color=MUTED, ls="-.", lw=1, label="starting LLM, mean pooled")
    ax.set_xscale("log", base=2)
    _style(ax, f"{label}: quality by vector size", "dimensions", "NanoBEIR nDCG@10")
    ax.legend(fontsize=8, frameon=False)
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
