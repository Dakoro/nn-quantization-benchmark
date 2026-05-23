"""
Étape 4 du pipeline — Benchmark latence, taille, mémoire.

Pour chaque modèle ONNX (FP32, INT8 dynamique, INT8 statique), mesure :
- Taille fichier sur disque
- RSS mémoire en 3 points (baseline / après chargement / pic pendant inférence)
- Latence d'inférence sur 200 runs (avec 20 warmup runs)
- Statistiques robustes : median, p95, p99, mean, std, min, max

Décisions de design :
- Threads = 1 par défaut : comportement single-thread, proche d'un déploiement
  embarqué et plus interprétable (on ne mesure pas la qualité du parallélisme ORT).
- Batch = 1 : cas standard d'inférence pour la classification.
- Médiane + percentiles plutôt que moyenne : robuste aux outliers OS.
- Input fixé par seed pour la reproductibilité.

Limitations honnêtes :
- Mesure mémoire single-process : les sessions ORT précédentes peuvent laisser
  des buffers qui gonflent légèrement les RSS suivantes. Pour un sizing
  production strict, un subprocess par modèle serait plus propre. Ici la
  tendance (FP32 vs INT8) reste largement dominante par rapport au bruit.
- Mesure CPU sur machine non-isolée : autres process et thermal throttling
  peuvent introduire du bruit. Les percentiles p95/p99 reflètent ce bruit.

Usage :
    python benchmark/run_benchmark.py
    python benchmark/run_benchmark.py --runs 500 --warmup 50
    python benchmark/run_benchmark.py --threads 4 --batch-size 8
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import platform
import sys
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnxruntime as ort
import psutil


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Modèles à benchmarker (par défaut, surchargeable via CLI)
# ---------------------------------------------------------------------------

DEFAULT_MODELS: list[tuple[str, str]] = [
    ("FP32",         "models/mobilenetv2_fp32.onnx"),
    ("INT8 dynamic", "models/mobilenetv2_int8_dynamic.onnx"),
    ("INT8 static",  "models/mobilenetv2_int8_static.onnx"),
]


# ---------------------------------------------------------------------------
# Dataclasses pour résultats sérialisables
# ---------------------------------------------------------------------------

@dataclass
class MemoryStats:
    baseline_mb: float
    after_load_mb: float
    peak_during_inference_mb: float
    delta_load_mb: float
    delta_peak_mb: float


@dataclass
class LatencyStats:
    n_runs: int
    n_warmup: int
    min_ms: float
    max_ms: float
    mean_ms: float
    median_ms: float
    std_ms: float
    p95_ms: float
    p99_ms: float


@dataclass
class ModelBenchmark:
    name: str
    path: str
    file_size_mb: float
    memory: MemoryStats
    latency: LatencyStats


@dataclass
class BenchmarkReport:
    metadata: dict
    results: list[ModelBenchmark] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Mesure
# ---------------------------------------------------------------------------

def get_rss_mb() -> float:
    """Renvoie la RSS du process courant en Mo."""
    return psutil.Process().memory_info().rss / (1024 ** 2)


def measure_model(
    name: str,
    path: Path,
    input_data: np.ndarray,
    n_runs: int,
    n_warmup: int,
    threads: int,
) -> ModelBenchmark:
    """Benchmark complet d'un modèle ONNX."""
    if not path.exists():
        raise FileNotFoundError(f"Modèle introuvable : {path}")

    logger.info("=" * 60)
    logger.info("Benchmark : %s (%s)", name, path)

    # Baseline mémoire : on force un gc.collect pour éliminer la pollution
    # des sessions précédentes (limitation single-process documentée).
    gc.collect()
    time.sleep(0.5)  # laisser le système se stabiliser
    baseline_rss = get_rss_mb()

    # Taille fichier
    file_size_mb = path.stat().st_size / (1024 ** 2)

    # Configuration ORT : threads contrôlés pour une mesure reproductible
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = threads
    sess_options.inter_op_num_threads = threads
    # GraphOptimizationLevel.ORT_ENABLE_ALL est le défaut, on le garde explicite
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(
        str(path),
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name

    # RSS après chargement
    after_load_rss = get_rss_mb()
    logger.info(
        "  taille fichier : %.2f Mo | RSS baseline -> après load : %.1f -> %.1f Mo (+%.1f)",
        file_size_mb, baseline_rss, after_load_rss, after_load_rss - baseline_rss,
    )

    # Warmup : on jette ces N premières inférences
    logger.info("  warmup : %d runs", n_warmup)
    for _ in range(n_warmup):
        session.run(None, {input_name: input_data})

    # Mesures : N runs avec timing + sampling RSS
    logger.info("  mesures : %d runs", n_runs)
    latencies_ms = np.empty(n_runs, dtype=np.float64)
    rss_samples = np.empty(n_runs, dtype=np.float64)

    for i in range(n_runs):
        t0 = time.perf_counter()
        session.run(None, {input_name: input_data})
        t1 = time.perf_counter()
        latencies_ms[i] = (t1 - t0) * 1000.0
        # Sample RSS toutes les 20 inférences (psutil overhead négligeable mais on évite)
        if i % 20 == 0:
            rss_samples[i] = get_rss_mb()
        else:
            rss_samples[i] = rss_samples[i - 1]

    peak_rss = float(rss_samples.max())

    # Statistiques
    latency = LatencyStats(
        n_runs=n_runs,
        n_warmup=n_warmup,
        min_ms=float(latencies_ms.min()),
        max_ms=float(latencies_ms.max()),
        mean_ms=float(latencies_ms.mean()),
        median_ms=float(np.median(latencies_ms)),
        std_ms=float(latencies_ms.std()),
        p95_ms=float(np.percentile(latencies_ms, 95)),
        p99_ms=float(np.percentile(latencies_ms, 99)),
    )
    memory = MemoryStats(
        baseline_mb=baseline_rss,
        after_load_mb=after_load_rss,
        peak_during_inference_mb=peak_rss,
        delta_load_mb=after_load_rss - baseline_rss,
        delta_peak_mb=peak_rss - baseline_rss,
    )

    logger.info(
        "  latence ms : median=%.2f  p95=%.2f  p99=%.2f  mean±std=%.2f±%.2f",
        latency.median_ms, latency.p95_ms, latency.p99_ms,
        latency.mean_ms, latency.std_ms,
    )
    logger.info(
        "  mémoire Mo : load +%.1f, pic +%.1f",
        memory.delta_load_mb, memory.delta_peak_mb,
    )

    # Libération explicite de la session avant le modèle suivant
    del session
    gc.collect()

    return ModelBenchmark(
        name=name,
        path=str(path),
        file_size_mb=file_size_mb,
        memory=memory,
        latency=latency,
    )


# ---------------------------------------------------------------------------
# Affichage tableau (sans dépendance externe, format markdown-compatible)
# ---------------------------------------------------------------------------

def format_comparison_table(results: list[ModelBenchmark]) -> str:
    """Formate un tableau ASCII/Markdown comparatif lisible directement."""
    if not results:
        return "(aucun résultat)"

    fp32 = next((r for r in results if r.name == "FP32"), results[0])
    ref_latency = fp32.latency.median_ms
    ref_size = fp32.file_size_mb

    headers = [
        "Format",
        "Taille (Mo)",
        "vs FP32",
        "Lat. p50 (ms)",
        "Lat. p95 (ms)",
        "Speedup",
        "RAM pic (Mo)",
    ]
    rows: list[list[str]] = []
    for r in results:
        size_ratio = ref_size / r.file_size_mb if r.file_size_mb else float("inf")
        speedup = ref_latency / r.latency.median_ms if r.latency.median_ms else float("inf")
        rows.append([
            r.name,
            f"{r.file_size_mb:.2f}",
            f"{size_ratio:.2f}x" if r is not fp32 else "—",
            f"{r.latency.median_ms:.2f}",
            f"{r.latency.p95_ms:.2f}",
            f"{speedup:.2f}x" if r is not fp32 else "—",
            f"{r.memory.delta_peak_mb:.1f}",
        ])

    # Calcul largeurs colonnes
    widths = [max(len(h), *(len(row[i]) for row in rows)) for i, h in enumerate(headers)]

    def fmt(row: list[str]) -> str:
        return "| " + " | ".join(c.ljust(w) for c, w in zip(row, widths)) + " |"

    sep = "|" + "|".join("-" * (w + 2) for w in widths) + "|"

    lines = [fmt(headers), sep, *(fmt(row) for row in rows)]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Métadonnées système
# ---------------------------------------------------------------------------

def collect_metadata(n_runs: int, n_warmup: int, batch_size: int, threads: int) -> dict:
    """Collecte les infos système pour reproductibilité."""
    try:
        cpu_info = platform.processor() or "unknown"
        # /proc/cpuinfo sur Linux donne souvent un nom plus précis
        if platform.system() == "Linux":
            cpuinfo = Path("/proc/cpuinfo")
            if cpuinfo.exists():
                for line in cpuinfo.read_text().splitlines():
                    if line.startswith("model name"):
                        cpu_info = line.split(":", 1)[1].strip()
                        break
    except Exception:
        cpu_info = "unknown"

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": f"{platform.system()} {platform.release()} {platform.machine()}",
        "python_version": sys.version.split()[0],
        "cpu": cpu_info,
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "total_ram_gb": round(psutil.virtual_memory().total / 1024**3, 1),
        "onnxruntime_version": ort.__version__,
        "numpy_version": np.__version__,
        "config": {
            "n_inference_runs": n_runs,
            "n_warmup_runs": n_warmup,
            "batch_size": batch_size,
            "threads": threads,
            "provider": "CPUExecutionProvider",
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark latence/taille/mémoire des modèles ONNX")
    parser.add_argument("--runs", type=int, default=200, help="Nombre d'inférences mesurées (défaut: 200)")
    parser.add_argument("--warmup", type=int, default=20, help="Nombre d'inférences de warmup (défaut: 20)")
    parser.add_argument("--batch-size", type=int, default=1, help="Taille de batch (défaut: 1)")
    parser.add_argument("--threads", type=int, default=1,
                        help="Threads ORT intra/inter-op (défaut: 1, pour reproductibilité). Mettre 0 = défaut ORT.")
    parser.add_argument("--seed", type=int, default=42, help="Seed numpy pour input (défaut: 42)")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/benchmark_results.json"),
        help="Fichier JSON de sortie",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("models"),
        help="Dossier racine des modèles (pour les chemins relatifs)",
    )
    args = parser.parse_args()

    # Prepare input
    rng = np.random.default_rng(args.seed)
    input_data = rng.standard_normal((args.batch_size, 3, 224, 224), dtype=np.float32)

    metadata = collect_metadata(args.runs, args.warmup, args.batch_size, args.threads)
    logger.info("Système : %s | %s | %d cœurs logiques | %.0f Go RAM",
                metadata["platform"], metadata["cpu"],
                metadata["cpu_count_logical"], metadata["total_ram_gb"])
    logger.info("ONNX Runtime %s, threads=%d, batch=%d, runs=%d, warmup=%d",
                metadata["onnxruntime_version"], args.threads,
                args.batch_size, args.runs, args.warmup)

    # Benchmark chaque modèle
    results: list[ModelBenchmark] = []
    for name, rel_path in DEFAULT_MODELS:
        path = Path(rel_path)
        if not path.exists():
            logger.warning("Modèle introuvable, skip : %s", path)
            continue
        try:
            result = measure_model(
                name=name,
                path=path,
                input_data=input_data,
                n_runs=args.runs,
                n_warmup=args.warmup,
                threads=args.threads,
            )
            results.append(result)
        except Exception as e:
            logger.error("Échec benchmark %s : %s", name, e)

    if not results:
        logger.error("Aucun résultat collecté. Vérifie que models/*.onnx existent.")
        sys.exit(1)

    # Persistance JSON
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = BenchmarkReport(
        metadata=metadata,
        results=results,
    )
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2, ensure_ascii=False)
    logger.info("Résultats écrits dans %s", args.output)

    # Affichage tableau comparatif
    print()
    print("Comparaison (référence: FP32) :")
    print()
    print(format_comparison_table(results))
    print()


if __name__ == "__main__":
    main()