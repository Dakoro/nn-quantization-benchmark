"""
Étape 6 du pipeline — Analyse et visualisation des résultats.

Lit les deux JSON produits par run_benchmark.py et evaluate_accuracy.py,
puis génère 4 figures matplotlib (.png) dans results/figures/ :

  01_overview.png        — Vue 3-panneaux : taille / latence / accuracy
  02_tradeoff.png        — Scatter compression × accuracy retention, taille = latence relative
  03_per_class.png       — Heatmap accuracy par classe Imagenette × format
  04_memory_profile.png  — Profils RSS mémoire (baseline / load / peak) par format

Plus un tableau de synthèse Markdown imprimé sur stdout, prêt à coller
dans le README.

Format script (.py) plutôt que notebook (.ipynb) :
- Versionnable proprement dans Git (diff lisible)
- Exécutable directement (`uv run notebooks/results_analysis.py`)
- Convertible en notebook via jupytext si besoin

Usage :
    python notebooks/results_analysis.py
    python notebooks/results_analysis.py --no-display  # sans plt.show()
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Palette colors : bleu calme / orange vif (à éviter) / vert positif
COLORS = {
    "FP32": "#4C72B0",
    "INT8 dynamic": "#DD8452",
    "INT8 static": "#55A868",
}

FORMAT_ORDER = ["FP32", "INT8 dynamic", "INT8 static"]


def load_results(bench_path: Path, acc_path: Path) -> tuple[dict, dict]:
    """Charge les deux JSON et indexe les résultats par nom de format."""
    bench = json.loads(bench_path.read_text())
    acc = json.loads(acc_path.read_text())

    bench_by_name = {r["name"]: r for r in bench["results"]}
    acc_by_name = {r["name"]: r for r in acc["results"]}

    return {"metadata": bench["metadata"], "by_name": bench_by_name}, \
           {"metadata": acc["metadata"], "by_name": acc_by_name}


def _formats_present(bench: dict, acc: dict) -> list[str]:
    """Renvoie les formats présents dans les deux JSON, dans l'ordre canonique."""
    return [f for f in FORMAT_ORDER
            if f in bench["by_name"] and f in acc["by_name"]]


# ---------------------------------------------------------------------------
# Figure 1 — Vue d'ensemble 3 panneaux
# ---------------------------------------------------------------------------

def fig_overview(bench: dict, acc: dict, out_path: Path) -> None:
    formats = _formats_present(bench, acc)
    sizes = [bench["by_name"][f]["file_size_mb"] for f in formats]
    lat_p50 = [bench["by_name"][f]["latency"]["median_ms"] for f in formats]
    lat_p95 = [bench["by_name"][f]["latency"]["p95_ms"] for f in formats]
    accuracies = [acc["by_name"][f]["imagenet_top1_acc"] * 100 for f in formats]
    colors = [COLORS[f] for f in formats]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5))

    # Panneau A — Taille
    bars = axes[0].bar(formats, sizes, color=colors, edgecolor="black", linewidth=0.5)
    axes[0].set_ylabel("Taille (Mo)")
    axes[0].set_title("Taille fichier ONNX")
    axes[0].grid(axis="y", alpha=0.3)
    for bar, v in zip(bars, sizes):
        axes[0].text(bar.get_x() + bar.get_width() / 2, v + max(sizes) * 0.015,
                     f"{v:.2f}", ha="center", fontsize=10)

    # Panneau B — Latence (log scale pour rendre INT8 dynamic lisible)
    bars = axes[1].bar(formats, lat_p50, color=colors, edgecolor="black", linewidth=0.5)
    # Error bar p50 -> p95
    yerr = [[0] * len(formats), [p95 - p50 for p95, p50 in zip(lat_p95, lat_p50)]]
    axes[1].errorbar(range(len(formats)), lat_p50, yerr=yerr,
                     fmt="none", ecolor="black", capsize=4, alpha=0.6)
    axes[1].set_ylabel("Latence (ms, échelle log)")
    axes[1].set_title("Latence d'inférence (p50, barre = p50→p95)")
    axes[1].set_yscale("log")
    axes[1].grid(axis="y", alpha=0.3, which="both")
    for bar, v in zip(bars, lat_p50):
        axes[1].text(bar.get_x() + bar.get_width() / 2, v * 1.15,
                     f"{v:.2f} ms", ha="center", fontsize=10)

    # Panneau C — Accuracy
    bars = axes[2].bar(formats, accuracies, color=colors, edgecolor="black", linewidth=0.5)
    axes[2].set_ylabel("ImageNet Top-1 (%)")
    axes[2].set_title("Précision sur Imagenette val")
    axes[2].set_ylim(min(accuracies) - 5, max(accuracies) + 3)
    axes[2].grid(axis="y", alpha=0.3)
    for bar, v in zip(bars, accuracies):
        axes[2].text(bar.get_x() + bar.get_width() / 2, v + 0.3,
                     f"{v:.2f}%", ha="center", fontsize=10)

    plt.suptitle("Comparaison MobileNetV2 : FP32 vs INT8 quantizations",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  → {out_path}")


# ---------------------------------------------------------------------------
# Figure 2 — Tradeoff scatter compression × accuracy retention
# ---------------------------------------------------------------------------

def fig_tradeoff(bench: dict, acc: dict, out_path: Path) -> None:
    formats = _formats_present(bench, acc)
    fp32_size = bench["by_name"]["FP32"]["file_size_mb"]
    fp32_lat = bench["by_name"]["FP32"]["latency"]["median_ms"]
    fp32_acc = acc["by_name"]["FP32"]["imagenet_top1_acc"]

    compressions = [fp32_size / bench["by_name"][f]["file_size_mb"] for f in formats]
    accuracy_retention = [acc["by_name"][f]["imagenet_top1_acc"] / fp32_acc * 100
                          for f in formats]
    speedups = [fp32_lat / bench["by_name"][f]["latency"]["median_ms"] for f in formats]
    # Inverse pour la taille (plus lent = plus petit point pour visibilité)
    # On utilise speedup directement pour la taille (plus rapide = plus gros)
    sizes_pts = [max(speedup, 0.1) * 800 for speedup in speedups]

    fig, ax = plt.subplots(figsize=(9, 6))
    for i, f in enumerate(formats):
        ax.scatter(
            compressions[i], accuracy_retention[i],
            s=sizes_pts[i], c=COLORS[f], alpha=0.7,
            edgecolors="black", linewidth=1, label=f,
        )
        ax.annotate(
            f"{f}\nspeedup ×{speedups[i]:.2f}\nlatence {bench['by_name'][f]['latency']['median_ms']:.2f} ms",
            xy=(compressions[i], accuracy_retention[i]),
            xytext=(15, 10), textcoords="offset points",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor=COLORS[f]),
        )

    ax.set_xlabel("Compression vs FP32 (×)")
    ax.set_ylabel("Accuracy retention vs FP32 (%)")
    ax.set_title("Tradeoff Compression × Accuracy (taille point ∝ speedup latence)")
    ax.axhline(100, color="gray", linestyle="--", alpha=0.5, label="iso-accuracy")
    ax.axvline(1, color="gray", linestyle=":", alpha=0.3)
    ax.set_xlim(0.5, max(compressions) * 1.15)
    ax.set_ylim(min(accuracy_retention) - 2, max(accuracy_retention) + 2)
    ax.grid(alpha=0.3)

    # Annotation explicative
    ax.text(
        0.98, 0.02,
        "Idéal : haut-droite (compression haute + accuracy haute)\nGros point = inférence rapide",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=9, alpha=0.7,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.5),
    )

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  → {out_path}")


# ---------------------------------------------------------------------------
# Figure 3 — Heatmap per-class
# ---------------------------------------------------------------------------

def fig_per_class_heatmap(acc: dict, out_path: Path) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from benchmark.imagenette import IMAGENETTE_CLASSES

    formats = [f for f in FORMAT_ORDER if f in acc["by_name"]]
    wnids = list(IMAGENETTE_CLASSES.keys())
    labels = [IMAGENETTE_CLASSES[w][1] for w in wnids]

    # Matrix (n_classes, n_formats)
    matrix = np.zeros((len(wnids), len(formats)))
    for j, f in enumerate(formats):
        per_class = acc["by_name"][f]["per_class_top1_acc"]
        for i, w in enumerate(wnids):
            matrix[i, j] = per_class.get(w, 0.0) * 100

    fig, ax = plt.subplots(figsize=(7, 6.5))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=30, vmax=100)

    ax.set_xticks(range(len(formats)))
    ax.set_xticklabels(formats, rotation=0)
    ax.set_yticks(range(len(wnids)))
    ax.set_yticklabels([f"{lbl}\n({w})" for w, lbl in zip(wnids, labels)],
                       fontsize=9)

    # Annotations
    for i in range(len(wnids)):
        for j in range(len(formats)):
            color = "white" if matrix[i, j] < 60 else "black"
            ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center",
                    color=color, fontsize=10, fontweight="bold")

    ax.set_title("Top-1 accuracy par classe Imagenette × format (%)")
    plt.colorbar(im, ax=ax, label="Top-1 (%)")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  → {out_path}")


# ---------------------------------------------------------------------------
# Figure 4 — Memory profile
# ---------------------------------------------------------------------------

def fig_memory_profile(bench: dict, out_path: Path) -> None:
    formats = [f for f in FORMAT_ORDER if f in bench["by_name"]]
    baselines = [bench["by_name"][f]["memory"]["baseline_mb"] for f in formats]
    after_loads = [bench["by_name"][f]["memory"]["after_load_mb"] for f in formats]
    peaks = [bench["by_name"][f]["memory"]["peak_during_inference_mb"] for f in formats]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel A — RSS absolu (baseline / load / peak)
    x = np.arange(len(formats))
    width = 0.27
    axes[0].bar(x - width, baselines, width, label="Baseline", color="#9DA0A8")
    axes[0].bar(x, after_loads, width, label="Après chargement", color="#7AB1DD")
    axes[0].bar(x + width, peaks, width, label="Pic inférence", color="#E07A5F")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(formats)
    axes[0].set_ylabel("RSS (Mo)")
    axes[0].set_title("Empreinte mémoire RSS (absolue)")
    axes[0].legend(loc="upper left")
    axes[0].grid(axis="y", alpha=0.3)

    # Panel B — Delta peak vs baseline (la métrique qui compte pour l'embarqué)
    deltas = [bench["by_name"][f]["memory"]["delta_peak_mb"] for f in formats]
    colors = [COLORS[f] for f in formats]
    bars = axes[1].bar(formats, deltas, color=colors, edgecolor="black", linewidth=0.5)
    axes[1].set_ylabel("Δ RSS pic − baseline (Mo)")
    axes[1].set_title("Mémoire pic d'inférence (vs baseline du process)")
    axes[1].grid(axis="y", alpha=0.3)
    for bar, v in zip(bars, deltas):
        axes[1].text(bar.get_x() + bar.get_width() / 2,
                     v + max(deltas) * 0.02,
                     f"{v:.1f} Mo", ha="center", fontsize=10)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  → {out_path}")


# ---------------------------------------------------------------------------
# Tableau de synthèse (Markdown, stdout)
# ---------------------------------------------------------------------------

def print_synthesis_table(bench: dict, acc: dict) -> None:
    formats = _formats_present(bench, acc)
    fp32 = bench["by_name"]["FP32"]
    fp32_acc_b = acc["by_name"]["FP32"]

    print()
    print("## Tableau de synthèse (à coller dans le README)")
    print()
    print("| Format | Taille (Mo) | Compression | Lat. p50 (ms) | Speedup | RAM pic (Mo) | Top-1 (%) | Δ Top-1 (pp) |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for f in formats:
        b = bench["by_name"][f]
        a = acc["by_name"][f]
        size = b["file_size_mb"]
        compression = fp32["file_size_mb"] / size
        lat = b["latency"]["median_ms"]
        speedup = fp32["latency"]["median_ms"] / lat
        mem = b["memory"]["delta_peak_mb"]
        top1 = a["imagenet_top1_acc"] * 100
        delta = (a["imagenet_top1_acc"] - fp32_acc_b["imagenet_top1_acc"]) * 100
        is_fp32 = (f == "FP32")
        print(f"| {f} | {size:.2f} | {'—' if is_fp32 else f'{compression:.2f}×'} | "
              f"{lat:.2f} | {'—' if is_fp32 else f'{speedup:.2f}×'} | "
              f"{mem:.1f} | {top1:.2f} | {'—' if is_fp32 else f'{delta:+.2f}'} |")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse et visualisation des résultats")
    parser.add_argument("--bench", type=Path, default=Path("results/benchmark_results.json"))
    parser.add_argument("--acc", type=Path, default=Path("results/accuracy_results.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/figures"))
    parser.add_argument("--no-display", action="store_true", help="N'affiche pas les figures")
    args = parser.parse_args()

    if not args.bench.exists():
        sys.exit(f"JSON benchmark introuvable : {args.bench}. "
                 f"Lance d'abord benchmark/run_benchmark.py")
    if not args.acc.exists():
        sys.exit(f"JSON accuracy introuvable : {args.acc}. "
                 f"Lance d'abord benchmark/evaluate_accuracy.py")

    bench, acc = load_results(args.bench, args.acc)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Génération des figures dans {args.output_dir}/ :")
    fig_overview(bench, acc, args.output_dir / "01_overview.png")
    fig_tradeoff(bench, acc, args.output_dir / "02_tradeoff.png")
    fig_per_class_heatmap(acc, args.output_dir / "03_per_class.png")
    fig_memory_profile(bench, args.output_dir / "04_memory_profile.png")

    print_synthesis_table(bench, acc)

    if not args.no_display:
        plt.show()


if __name__ == "__main__":
    main()