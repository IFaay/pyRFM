# compare_logs.py
# -*- coding: utf-8 -*-
import re
import csv
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

EPOCH_RE = re.compile(
    r"^Epoch\s*\[(?P<epoch>\d+)\s*/\s*(?P<total>\d+)\]\s*"
    r"Loss:\s*(?P<loss>[+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*\|\s*"
    r"Best:\s*(?P<best>[+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*\|\s*"
    r"LR:\s*(?P<lr>[+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*\|\s*"
    r"ValAngle\(deg\):\s*mean=(?P<val_mean>[+-]?\d+(?:\.\d+)?),\s*max=(?P<val_max>[+-]?\d+(?:\.\d+)?)",
    re.IGNORECASE
)
RESET_RE = re.compile(r"^\[Clean\].*tag=(?P<tag>[^.\]]+)", re.IGNORECASE)


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def set_paper_rc(latex=False, width=3.8, height=2.2, dpi=300):
    """顶会友好风格 (Times 字体 / 粗线 / 合适尺寸)"""
    plt.rcParams.update({
        "figure.figsize": (width, height),
        "figure.dpi": dpi,
        "savefig.dpi": dpi,
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.labelsize": 10.5,
        "xtick.labelsize": 9.0,
        "ytick.labelsize": 9.0,
        "legend.fontsize": 9.0,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.6,
        "lines.markeredgewidth": 0.8,
        "lines.markersize": 3.8,
        "grid.linewidth": 0.6,
        "grid.alpha": 0.3,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.linestyle": "--",
        "legend.frameon": False,
    })
    if latex:
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "CMU Serif"],
        })
    else:
        plt.rcParams.update({
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        })


def parse_log(path: Path):
    epochs, loss, best, lr, val_mean, val_max = [], [], [], [], [], []
    tag = None
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            mtag = RESET_RE.match(line)
            if mtag and not tag:
                tag = mtag.group("tag").strip()
            m = EPOCH_RE.match(line)
            if not m:
                continue
            ep = int(m.group("epoch"))
            epochs.append(ep)
            loss.append(safe_float(m.group("loss")))
            best.append(safe_float(m.group("best")))
            lr.append(safe_float(m.group("lr")))
            val_mean.append(safe_float(m.group("val_mean")))
            val_max.append(safe_float(m.group("val_max")))
    if not tag:
        tag = path.stem
    return {
        "label": tag,
        "epochs": np.asarray(epochs, dtype=int),
        "loss": np.asarray(loss, dtype=float),
        "best": np.asarray(best, dtype=float),
        "lr": np.asarray(lr, dtype=float),
        "val_mean": np.asarray(val_mean, dtype=float),
        "val_max": np.asarray(val_max, dtype=float),
    }


def save_csv(data, out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        import csv
        w = csv.writer(f)
        w.writerow(["epoch", "loss", "best", "lr", "val_mean", "val_max"])
        for i in range(len(data["epochs"])):
            w.writerow([
                int(data["epochs"][i]),
                data["loss"][i],
                data["best"][i],
                data["lr"][i],
                data["val_mean"][i],
                data["val_max"][i],
            ])


def plot_compare(d1, d2, key, ylabel, out_base: Path, yscale="log",
                 legend_outside=False, fmt_list=("pdf", "png")):
    x1, y1 = d1["epochs"], d1[key]
    x2, y2 = d2["epochs"], d2[key]

    fig, ax = plt.subplots(dpi=600)

    ax.plot(x1, y1, label=d1["label"], linestyle="-", marker='o', markevery=max(1, len(x1) // 20), markersize=4)
    ax.plot(x2, y2, label=d2["label"], linestyle="--", marker='s', markevery=max(1, len(x2) // 20), markersize=4)

    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    if yscale:
        ax.set_yscale(yscale)

    ax.minorticks_on()

    if legend_outside:
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
        fig.tight_layout(pad=0.6)
    else:
        ax.legend()
        fig.tight_layout()

    out_base.parent.mkdir(parents=True, exist_ok=True)
    for ext in fmt_list:
        fig.savefig(out_base.with_suffix(f".{ext}"))
    plt.close(fig)


def main():
    DEFAULT_LOG1 = Path("logs/relu-tanh/train_20250915_000207.log")
    DEFAULT_LOG2 = Path("logs/tanh-tanh/train_20250915_101247.log")
    DEFAULT_OUT = Path("compare_outputs")

    ap = argparse.ArgumentParser(description="Compare two training logs (publication-quality plots).")
    ap.add_argument("logfile1", type=str, nargs="?", default=str(DEFAULT_LOG1))
    ap.add_argument("logfile2", type=str, nargs="?", default=str(DEFAULT_LOG2))
    ap.add_argument("--outdir", type=str, default=str(DEFAULT_OUT))
    ap.add_argument("--labels", type=str, default=None)
    ap.add_argument("--yscale", type=str, choices=["linear", "log", "symlog"], default="log")
    ap.add_argument("--latex", action="store_true")
    ap.add_argument("--format", type=str, default="pdf,png")
    args = ap.parse_args()

    set_paper_rc(latex=args.latex)

    p1, p2 = Path(args.logfile1), Path(args.logfile2)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    d1 = parse_log(p1)
    d2 = parse_log(p2)

    if args.labels:
        parts = [s.strip() for s in args.labels.split(",")]
        if len(parts) == 2:
            d1["label"], d2["label"] = parts[0], parts[1]

    save_csv(d1, outdir / f"metrics_{d1['label']}.csv")
    save_csv(d2, outdir / f"metrics_{d2['label']}.csv")

    fmt_list = tuple([s.strip() for s in args.format.split(",") if s.strip() in ("pdf", "png", "svg")])

    plot_compare(d1, d2, "loss", "Training Loss", outdir / "loss_compare", yscale=args.yscale, fmt_list=fmt_list)
    plot_compare(d1, d2, "best", "Best (running)", outdir / "best_compare", yscale=args.yscale, fmt_list=fmt_list)
    plot_compare(d1, d2, "val_mean", r"ValAngle Mean (deg)", outdir / "val_mean_compare",
                 yscale="linear" if args.yscale == "log" else args.yscale, fmt_list=fmt_list)
    plot_compare(d1, d2, "val_max", r"ValAngle Max (deg)", outdir / "val_max_compare",
                 yscale="linear" if args.yscale == "log" else args.yscale, fmt_list=fmt_list)
    plot_compare(d1, d2, "lr", "Learning Rate", outdir / "lr_compare", yscale=args.yscale, fmt_list=fmt_list)

    print(f"完成。图与 CSV 已保存到: {outdir.resolve()}")
    print(f"曲线标签：{d1['label']} vs {d2['label']}")


if __name__ == "__main__":
    main()
