"""
Étape 5 du pipeline — Évaluation Top-1 et Top-5 sur Imagenette val.

Pour chaque modèle ONNX (FP32, INT8 dynamic, INT8 static), mesure la précision
de classification sur le split validation d'Imagenette (~3925 images).

Deux schémas de scoring complémentaires :

1. ImageNet-restricted Top-1 / Top-5 (métrique principale, paper-grade) :
   On prend l'argmax sur les 1000 logits ImageNet. On compte comme correct
   uniquement si l'index prédit correspond exactement à la classe Imagenette
   attendue (parmi {0, 217, 482, 491, 497, 566, 569, 571, 574, 701}).
   Baseline aléatoire : 0.1%. C'est la métrique difficile, celle qui expose
   le drop de quantification.

2. Imagenette-masked Top-1 (métrique secondaire, démonstration) :
   On ne regarde que les 10 logits correspondant aux classes Imagenette, on
   prend l'argmax sur ce subset. Baseline aléatoire : 10%. C'est plus parlant
   visuellement mais cache une partie du drop de quantif (les confusions vers
   des classes non-Imagenette sont effacées).

Décisions de design :
- Threads = 1 par défaut pour la cohérence avec le benchmark de latence
- Batch configurable (défaut 32) : on profite de l'axe batch dynamique de l'ONNX
- Preprocessing strictement équivalent à torchvision (cf. benchmark/imagenette.preprocess_image)
- Sortie : JSON résultats + tableau ASCII

Usage :
    python benchmark/evaluate_accuracy.py
    python benchmark/evaluate_accuracy.py --max-images 500  # smoke test rapide
    python benchmark/evaluate_accuracy.py --val-dir data/imagenette2-320/val --batch-size 64
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnxruntime as ort

# Import depuis le module compagnon. Lancer le script depuis la racine du projet
# (nn-quantization-benchmark/) pour que ce import fonctionne.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from benchmark.imagenette import (
    preprocess_image,
    IMAGENETTE_IMAGENET_INDICES,
    list_images_with_labels,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


DEFAULT_MODELS: list[tuple[str, str]] = [
    ("FP32",         "models/mobilenetv2_fp32.onnx"),
    ("INT8 dynamic", "models/mobilenetv2_int8_dynamic.onnx"),
    ("INT8 static",  "models/mobilenetv2_int8_static.onnx"),
]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class AccuracyResult:
    name: str
    path: str
    n_images: int
    # ImageNet-restricted (argmax sur 1000 classes)
    imagenet_top1_acc: float
    imagenet_top5_acc: float
    # Imagenette-masked (argmax sur 10 logits sélectionnés)
    imagenette_masked_top1_acc: float
    # Confusion par classe (utile pour analyse fine, top-1 ImageNet-restricted)
    per_class_top1_acc: dict[str, float]


@dataclass
class AccuracyReport:
    metadata: dict
    results: list[AccuracyResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Évaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    name: str,
    path: Path,
    images_with_labels: list[tuple[Path, int, str]],
    batch_size: int,
    threads: int,
) -> AccuracyResult:
    """Charge un modèle ONNX et évalue Top-1/Top-5 sur le set fourni."""
    if not path.exists():
        raise FileNotFoundError(f"Modèle introuvable : {path}")

    logger.info("=" * 60)
    logger.info("Évaluation : %s (%s)", name, path)

    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = threads
    sess_options.inter_op_num_threads = threads
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(
        str(path),
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name

    n_total = len(images_with_labels)
    imagenet_top1_correct = 0
    imagenet_top5_correct = 0
    imagenette_masked_top1_correct = 0

    # Compteurs par classe pour confusion fine
    per_class_total: dict[str, int] = {}
    per_class_top1: dict[str, int] = {}

    # Numpy array des indices Imagenette pour subset masking
    imagenette_indices = np.array(IMAGENETTE_IMAGENET_INDICES, dtype=np.int64)

    # Itération par batches
    for batch_start in range(0, n_total, batch_size):
        batch = images_with_labels[batch_start : batch_start + batch_size]

        # Preprocess (séquentiel ; pourrait être parallélisé)
        batch_inputs = np.stack([preprocess_image(p) for p, _, _ in batch])
        # Inférence batchée
        logits = session.run(None, {input_name: batch_inputs})[0]  # (B, 1000)

        # ImageNet-restricted Top-1 / Top-5
        top1_pred = logits.argmax(axis=1)
        # argpartition pour Top-5 (plus rapide que argsort sur 1000)
        top5_pred = np.argpartition(logits, -5, axis=1)[:, -5:]

        for i, (_, true_idx, wnid) in enumerate(batch):
            # ImageNet-restricted
            if top1_pred[i] == true_idx:
                imagenet_top1_correct += 1
                per_class_top1[wnid] = per_class_top1.get(wnid, 0) + 1
            if true_idx in top5_pred[i]:
                imagenet_top5_correct += 1
            per_class_total[wnid] = per_class_total.get(wnid, 0) + 1

            # Imagenette-masked : argmax restreint aux 10 indices Imagenette
            masked_logits = logits[i, imagenette_indices]  # (10,)
            masked_top1_idx = imagenette_indices[masked_logits.argmax()]
            if masked_top1_idx == true_idx:
                imagenette_masked_top1_correct += 1

        # Progress logging
        if (batch_start // batch_size) % 20 == 0:
            done = batch_start + len(batch)
            logger.info(
                "  %d/%d  (top1=%.3f, top5=%.3f, masked=%.3f)",
                done, n_total,
                imagenet_top1_correct / done,
                imagenet_top5_correct / done,
                imagenette_masked_top1_correct / done,
            )

    # Stats finales
    imagenet_top1_acc = imagenet_top1_correct / n_total
    imagenet_top5_acc = imagenet_top5_correct / n_total
    imagenette_masked_top1_acc = imagenette_masked_top1_correct / n_total

    per_class_acc = {
        wnid: per_class_top1.get(wnid, 0) / per_class_total[wnid]
        for wnid in per_class_total
    }

    logger.info(
        "  FINAL  imagenet_top1=%.4f  imagenet_top5=%.4f  imagenette_masked=%.4f",
        imagenet_top1_acc, imagenet_top5_acc, imagenette_masked_top1_acc,
    )

    del session
    gc.collect()

    return AccuracyResult(
        name=name,
        path=str(path),
        n_images=n_total,
        imagenet_top1_acc=imagenet_top1_acc,
        imagenet_top5_acc=imagenet_top5_acc,
        imagenette_masked_top1_acc=imagenette_masked_top1_acc,
        per_class_top1_acc=per_class_acc,
    )


# ---------------------------------------------------------------------------
# Affichage tableau
# ---------------------------------------------------------------------------

def format_accuracy_table(results: list[AccuracyResult]) -> str:
    """Tableau comparatif Markdown-compatible."""
    if not results:
        return "(aucun résultat)"

    fp32 = next((r for r in results if r.name == "FP32"), results[0])
    ref_top1 = fp32.imagenet_top1_acc

    headers = [
        "Format",
        "ImageNet Top-1 (%)",
        "Δ vs FP32 (pp)",
        "ImageNet Top-5 (%)",
        "Imagenette-masked Top-1 (%)",
    ]
    rows: list[list[str]] = []
    for r in results:
        delta_pp = (r.imagenet_top1_acc - ref_top1) * 100
        rows.append([
            r.name,
            f"{r.imagenet_top1_acc * 100:.2f}",
            f"{delta_pp:+.2f}" if r is not fp32 else "—",
            f"{r.imagenet_top5_acc * 100:.2f}",
            f"{r.imagenette_masked_top1_acc * 100:.2f}",
        ])

    widths = [max(len(h), *(len(row[i]) for row in rows)) for i, h in enumerate(headers)]

    def fmt(row: list[str]) -> str:
        return "| " + " | ".join(c.ljust(w) for c, w in zip(row, widths)) + " |"

    sep = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    return "\n".join([fmt(headers), sep, *(fmt(row) for row in rows)])


def format_per_class_table(results: list[AccuracyResult]) -> str:
    """Tableau de l'accuracy par classe (ImageNet-restricted Top-1)."""
    if not results:
        return ""
    from benchmark.imagenette import IMAGENETTE_CLASSES

    wnids = list(IMAGENETTE_CLASSES.keys())
    headers = ["Classe"] + [r.name for r in results]
    rows: list[list[str]] = []
    for wnid in wnids:
        _, label = IMAGENETTE_CLASSES[wnid]
        row = [f"{label} ({wnid})"]
        for r in results:
            acc = r.per_class_top1_acc.get(wnid, 0.0)
            row.append(f"{acc * 100:.2f}")
        rows.append(row)

    widths = [max(len(h), *(len(row[i]) for row in rows)) for i, h in enumerate(headers)]

    def fmt(row: list[str]) -> str:
        return "| " + " | ".join(c.ljust(w) for c, w in zip(row, widths)) + " |"

    sep = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    return "\n".join([fmt(headers), sep, *(fmt(row) for row in rows)])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Évalue Top-1/Top-5 des modèles ONNX sur Imagenette val"
    )
    parser.add_argument(
        "--val-dir",
        type=Path,
        default=Path("data/imagenette2-320/val"),
        help="Dossier val/ d'Imagenette",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Taille de batch pour l'inférence (défaut: 32)",
    )
    parser.add_argument(
        "--threads", type=int, default=1,
        help="Threads ORT (défaut: 1)",
    )
    parser.add_argument(
        "--max-images", type=int, default=None,
        help="Limite le nombre d'images évaluées (smoke test)",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/accuracy_results.json"),
        help="Fichier JSON de sortie",
    )
    args = parser.parse_args()

    # Lister les images du val + labels
    if not args.val_dir.exists():
        logger.error(
            "Dossier val introuvable : %s. "
            "As-tu lancé data/download_imagenette.py ?",
            args.val_dir,
        )
        sys.exit(1)

    logger.info("Listing des images dans %s ...", args.val_dir)
    images_with_labels = list_images_with_labels(args.val_dir)
    logger.info("Trouvé %d images sur %d classes",
                len(images_with_labels),
                len({wnid for _, _, wnid in images_with_labels}))

    if args.max_images is not None:
        images_with_labels = images_with_labels[: args.max_images]
        logger.info("Tronqué à %d images (--max-images)", len(images_with_labels))

    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "val_dir": str(args.val_dir),
        "n_images": len(images_with_labels),
        "batch_size": args.batch_size,
        "threads": args.threads,
        "onnxruntime_version": ort.__version__,
    }

    # Évaluation
    results: list[AccuracyResult] = []
    for name, rel_path in DEFAULT_MODELS:
        path = Path(rel_path)
        if not path.exists():
            logger.warning("Modèle introuvable, skip : %s", path)
            continue
        try:
            result = evaluate_model(
                name=name,
                path=path,
                images_with_labels=images_with_labels,
                batch_size=args.batch_size,
                threads=args.threads,
            )
            results.append(result)
        except Exception as e:
            logger.error("Échec évaluation %s : %s", name, e)

    if not results:
        logger.error("Aucun résultat. Vérifie que models/*.onnx existent.")
        sys.exit(1)

    # Sauvegarde JSON
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = AccuracyReport(metadata=metadata, results=results)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2, ensure_ascii=False)
    logger.info("Résultats écrits dans %s", args.output)

    # Tableaux
    print()
    print("=== Précision globale ===")
    print()
    print(format_accuracy_table(results))
    print()
    print("=== ImageNet-restricted Top-1 par classe (%) ===")
    print()
    print(format_per_class_table(results))
    print()


if __name__ == "__main__":
    main()