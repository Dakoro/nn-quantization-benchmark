"""
Analyse du scaling de latence en fonction du nombre de threads.

Prend en entrée plusieurs JSON benchmark (un par configuration de threads),
produit une figure à deux panneaux :
- Latence absolue par format vs threads (échelle log)
- Speedup parallèle vs 1 thread (= efficacité du multi-thread)

Usage :
    # Lance d'abord plusieurs benchmarks avec différents --threads :
    uv run benchmark/run_benchmark.py --threads 1 --output results/bench_1t.json
    uv run benchmark/run_benchmark.py --threads 4 --output results/bench_4t.json
    uv run benchmark/run_benchmark.py --threads 8 --output results/bench_8t.json

    # Puis génère la figure de scaling :
    uv run results/scaling_analysis.py \\
        results/bench_1t.json results/bench_4t.json results/bench_8t.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


COLORS = {
    "FP32": "#4C72B0",
    "INT8 dynamic": "#DD8452",
    "INT8 static": "#55A868",
}

MARKERS = {
    "FP32": "o",
    "INT8 dynamic": "s",
    "INT8 static": "^",
}


def load_benches(paths: list[Path]) -> list[tuple[int, dict]]:
    """Charge les JSONs, indexe par nombre de threads, trie."""
    out: list[tuple[int, dict]] = []
    for p in paths:
        data = json.loads(p.read_text())
        threads = data["metadata"]["config"]["threads"]
        out.append((threads, data))
    out.sort(key=lambda x: x[0])
    return out


def _lookup_latency(data: dict, format_name: str) -> float | None:
    """Renvoie la latence p50 d'un format dans un JSON benchmark."""
    for r in data["results"]:
        if r["name"] == format_name:
            return r["latency"]["median_ms"]
    return None


def plot_scaling(benches: list[tuple[int, dict]], out_path: Path) -> None:
    formats = ["FP32", "INT8 dynamic", "INT8 static"]
    thread_counts = [t for t, _ in benches]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5))

    # ---- Panneau A : latence absolue (log scale) ----
    for fmt in formats:
        latencies = [_lookup_latency(d, fmt) for _, d in benches]
        latencies = [lat for lat in latencies if lat is not None]
        threads_present = [t for t, d in benches if _lookup_latency(d, fmt) is not None]
        if not latencies:
            continue
        axes[0].plot(
            threads_present, latencies,
            marker=MARKERS[fmt], color=COLORS[fmt],
            linewidth=2, markersize=11, label=fmt,
        )
        # Annotations au-dessus de chaque point
        for t, lat in zip(threads_present, latencies):
            axes[0].annotate(
                f"{lat:.2f} ms", xy=(t, lat), xytext=(8, 6),
                textcoords="offset points", fontsize=8.5,
                color=COLORS[fmt],
            )

    axes[0].set_xlabel("Nombre de threads ORT")
    axes[0].set_ylabel("Latence p50 (ms, échelle log)")
    axes[0].set_yscale("log")
    axes[0].set_title("Latence d'inférence vs threads")
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.3, which="both")
    axes[0].set_xticks(thread_counts)

    # ---- Panneau B : speedup parallèle (vs 1 thread) ----
    max_threads = max(thread_counts)

    for fmt in formats:
        lat_1t = _lookup_latency(benches[0][1], fmt) if benches[0][0] == 1 else None
        if lat_1t is None:
            continue
        threads_present = []
        speedups = []
        for t, d in benches:
            lat = _lookup_latency(d, fmt)
            if lat is None:
                continue
            threads_present.append(t)
            speedups.append(lat_1t / lat)
        axes[1].plot(
            threads_present, speedups,
            marker=MARKERS[fmt], color=COLORS[fmt],
            linewidth=2, markersize=11, label=fmt,
        )
        for t, s in zip(threads_present, speedups):
            axes[1].annotate(
                f"×{s:.2f}", xy=(t, s), xytext=(8, 6),
                textcoords="offset points", fontsize=8.5,
                color=COLORS[fmt],
            )

    # Ligne de scaling idéal
    axes[1].plot([1, max_threads], [1, max_threads],
                 "k--", alpha=0.4, label="Scaling linéaire idéal", linewidth=1.5)

    axes[1].set_xlabel("Nombre de threads ORT")
    axes[1].set_ylabel("Speedup vs 1 thread")
    axes[1].set_title("Efficacité du parallélisme (1t → Nt)")
    axes[1].legend(loc="upper left")
    axes[1].grid(alpha=0.3)
    axes[1].set_xticks(thread_counts)

    plt.suptitle("Scaling multi-thread sur MobileNetV2",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  → {out_path}")


def print_scaling_table(benches: list[tuple[int, dict]]) -> None:
    """Affiche un tableau Markdown du scaling."""
    formats = ["FP32", "INT8 dynamic", "INT8 static"]
    thread_counts = [t for t, _ in benches]

    print()
    print("## Tableau de scaling (Markdown)")
    print()
    header = "| Format | " + " | ".join(f"{t}t (ms)" for t in thread_counts) + \
             " | Scaling 1t→{}t | Efficacité (%) |".format(thread_counts[-1])
    print(header)
    sep = "|" + "|".join(["---"] + ["---:"] * (len(thread_counts) + 2)) + "|"
    print(sep)

    for fmt in formats:
        latencies = [_lookup_latency(d, fmt) for _, d in benches]
        if any(lat is None for lat in latencies):
            continue
        cells = [f"{lat:.2f}" for lat in latencies]
        scaling = latencies[0] / latencies[-1]
        efficiency = scaling / thread_counts[-1] * 100
        print(f"| {fmt} | " + " | ".join(cells) +
              f" | ×{scaling:.2f} | {efficiency:.0f}% |")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyse du scaling multi-thread à partir de plusieurs JSONs benchmark"
    )
    parser.add_argument(
        "benchmark_jsons",
        type=Path,
        nargs="+",
        help="Liste de JSONs benchmark à différentes configs de threads",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/figures/05_scaling.png"),
        help="Fichier PNG de sortie",
    )
    args = parser.parse_args()

    benches = load_benches(args.benchmark_jsons)

    print(f"Chargé {len(benches)} benchmarks : "
          f"threads = {[t for t, _ in benches]}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plot_scaling(benches, args.output)
    print_scaling_table(benches)


if __name__ == "__main__":
    main()