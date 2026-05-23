"""
Étapes 2 et 3 du pipeline — Quantification INT8 de MobileNetV2 ONNX.

Deux modes :
- dynamic : quantifie les POIDS uniquement (Conv, MatMul, Gemm) en INT8.
            Les activations restent en FP32 et sont quantifiées à la volée
            au runtime. Pas besoin de jeu de calibration. Gain en taille
            ~4x, gain en latence modeste sur CPU x86.
- static  : quantifie POIDS + ACTIVATIONS. Nécessite un jeu de calibration
            (typiquement 100-512 images représentatives) pour estimer
            les distributions d'activations. Meilleur gain en latence,
            risque de drop d'accuracy plus important.

Pre-processing :
    Avant toute quantification, on applique onnxruntime.quantization.shape_inference
    .quant_pre_process qui fait :
    - Symbolic shape inference (résout les shapes dynamiques)
    - Optimization passes (fusion, constant folding renforcé)
    - ONNX shape inference standard
    C'est recommandé par Microsoft pour éviter des erreurs subtiles et de
    mauvaises performances après quantif. Cf. WARNING émis par onnxruntime
    quand on saute cette étape.

Schéma de quantification (static) :
- QuantFormat.QDQ (défaut) : insère des nœuds Quantize/Dequantize explicites.
  Format recommandé par Microsoft pour les architectures à depthwise
  convolutions (MobileNet, EfficientNet) car le chemin de code QOperator
  a un bug de re-quantification du bias qui crash en per-channel sur
  certains canaux à activations quasi-nulles. À l'inférence ORT, QDQ est
  optimisé vers les mêmes kernels INT8 que QOperator — la perf est
  équivalente. Alternative : --quant-format qoperator (peut crasher).
- per_channel=True : quantif par canal sur les Conv (1 scale/zero_point par
  canal de sortie). Meilleur en accuracy qu'une quantif per-tensor pour
  les architectures à depthwise comme MobileNet.
- weight_type=QInt8, activation_type=QUInt8 : combo standard CPU x86.
- calibrate_method=Percentile 99.999 (défaut) : écrête les 0.001% d'outliers
  d'activation. Plus robuste que MinMax qui peut être pollué par un seul
  outlier extrême et générer des scales catastrophiques.
- WeightSymmetric=True : force zero_point=0 pour les poids
  (scale = max(|min|, |max|)). Évite les ratios pathologiques entre canaux.

Usage :
    # Quantification dynamique (rapide, pas de données)
    python models/quantize.py --mode dynamic

    # Quantification statique (nécessite un dossier d'images)
    python models/quantize.py --mode static \\
        --calibration-dir data/imagenette2/val \\
        --calibration-size 512

    # Les deux d'un coup
    python models/quantize.py --mode both --calibration-dir data/imagenette2/val
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image

from onnxruntime.quantization import (
    quantize_dynamic,
    quantize_static,
    QuantType,
    QuantFormat,
    CalibrationDataReader,
    CalibrationMethod,
)
from onnxruntime.quantization.shape_inference import quant_pre_process


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ImageNet normalization standard (mêmes valeurs que torchvision.transforms)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Extensions d'images reconnues pour le calibration set
IMAGE_EXTS = {".jpeg", ".jpg", ".png", ".bmp"}


# ---------------------------------------------------------------------------
# Pre-processing image (équivalent torchvision : Resize(256) + CenterCrop(224))
# ---------------------------------------------------------------------------

def preprocess_image(img_path: Path) -> np.ndarray:
    """Charge, redimensionne, recadre et normalise une image pour MobileNetV2.

    Pipeline strictement équivalent à :
        T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    Retourne un tenseur (3, 224, 224) float32 en layout NCHW (sans la dim batch).
    """
    img = Image.open(img_path).convert("RGB")
    w, h = img.size

    # Resize : shorter side -> 256, en conservant le ratio
    if w < h:
        new_w, new_h = 256, int(round(h * 256 / w))
    else:
        new_w, new_h = int(round(w * 256 / h)), 256
    img = img.resize((new_w, new_h), Image.BILINEAR)

    # Center crop 224x224
    left = (new_w - 224) // 2
    top = (new_h - 224) // 2
    img = img.crop((left, top, left + 224, top + 224))

    # HWC uint8 -> CHW float32 normalisé
    arr = np.asarray(img, dtype=np.float32) / 255.0  # (H, W, 3)
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = arr.transpose(2, 0, 1)  # (3, H, W)
    return arr.astype(np.float32)


# ---------------------------------------------------------------------------
# CalibrationDataReader pour la quantification statique
# ---------------------------------------------------------------------------

class ImageFolderCalibrationReader(CalibrationDataReader):
    """Itère sur N images d'un dossier (récursif) pour la calibration statique.

    Agnostique au layout : fonctionne aussi bien sur :
    - ImageNet val : data/imagenet_val_subset/<class_id>/*.JPEG
    - Imagenette  : data/imagenette2/val/<class_name>/*.JPEG
    - Un dossier plat d'images quelconques

    Stratégie d'échantillonnage : on prend les N premières images (tri
    lexicographique) plutôt qu'aléatoire pour la reproductibilité.
    """

    def __init__(
        self,
        image_dir: Path,
        n_samples: int = 512,
        input_name: str = "input",
    ):
        self.image_dir = Path(image_dir)
        self.input_name = input_name

        if not self.image_dir.exists():
            raise FileNotFoundError(
                f"Dossier de calibration introuvable : {self.image_dir}"
            )

        all_images = sorted(
            p for p in self.image_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        )
        if not all_images:
            raise FileNotFoundError(
                f"Aucune image (.jpeg/.jpg/.png/.bmp) trouvée dans {self.image_dir}"
            )
        self.image_paths = all_images[:n_samples]
        logger.info(
            "Calibration set : %d / %d images depuis %s",
            len(self.image_paths), len(all_images), self.image_dir,
        )
        self._iter: Iterator[Path] = iter(self.image_paths)
        self._count = 0

    def get_next(self) -> dict | None:
        """Retourne le prochain batch (batch=1) ou None à la fin."""
        try:
            path = next(self._iter)
        except StopIteration:
            return None
        x = preprocess_image(path)[np.newaxis, ...]  # (1, 3, 224, 224)
        self._count += 1
        if self._count % 64 == 0:
            logger.info("  calibration : %d/%d", self._count, len(self.image_paths))
        return {self.input_name: x}

    def rewind(self) -> None:
        """Re-itère depuis le début (utile si calibrate_method en a besoin)."""
        self._iter = iter(self.image_paths)
        self._count = 0


# ---------------------------------------------------------------------------
# Quantification : modes dynamic et static
# ---------------------------------------------------------------------------

def preprocess_for_quantization(input_path: Path, output_path: Path) -> None:
    """Shape inference + optimisations avant quantification (recommandé par MS)."""
    logger.info("Pre-processing du modèle pour quantification : %s -> %s",
                input_path, output_path)
    quant_pre_process(
        input_model=str(input_path),
        output_model_path=str(output_path),
        skip_optimization=False,
        skip_onnx_shape=False,
        skip_symbolic_shape=False,
    )


def quantize_dynamic_int8(input_path: Path, output_path: Path) -> None:
    """Quantif dynamique INT8 : poids uniquement."""
    logger.info("Quantif DYNAMIQUE INT8 : %s -> %s", input_path, output_path)
    quantize_dynamic(
        model_input=str(input_path),
        model_output=str(output_path),
        weight_type=QuantType.QInt8,
    )


def quantize_static_int8(
    input_path: Path,
    output_path: Path,
    calibration_dir: Path,
    n_samples: int = 512,
    per_channel: bool = True,
    quant_format: QuantFormat = QuantFormat.QDQ,
    calibrate_method: CalibrationMethod = CalibrationMethod.Percentile,
    calib_percentile: float = 99.999,
    weight_symmetric: bool = True,
) -> None:
    """Quantif statique INT8 : poids + activations via calibration.

    Défauts robustes pour MobileNet (QDQ + Percentile + Symmetric weights),
    qui évitent le bug de re-quantification du bias en QOperator + MinMax.
    """
    logger.info(
        "Quantif STATIQUE INT8 : %s -> %s",
        input_path, output_path,
    )
    logger.info(
        "  format=%s, calibration=%s, per_channel=%s, weight_symmetric=%s, n=%d",
        quant_format.name, calibrate_method.name, per_channel, weight_symmetric, n_samples,
    )
    reader = ImageFolderCalibrationReader(
        image_dir=calibration_dir,
        n_samples=n_samples,
        input_name="input",
    )

    extra_options: dict = {
        "WeightSymmetric": weight_symmetric,
        "ActivationSymmetric": False,  # activations -> UInt8 asymmetric
    }
    if calibrate_method == CalibrationMethod.Percentile:
        extra_options["CalibPercentile"] = calib_percentile

    quantize_static(
        model_input=str(input_path),
        model_output=str(output_path),
        calibration_data_reader=reader,
        quant_format=quant_format,
        per_channel=per_channel,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8,
        calibrate_method=calibrate_method,
        extra_options=extra_options,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def log_size_comparison(original: Path, quantized: Path) -> None:
    """Affiche la comparaison de taille avant/après quantif."""
    orig_mb = original.stat().st_size / (1024 ** 2)
    quant_mb = quantized.stat().st_size / (1024 ** 2)
    ratio = orig_mb / quant_mb if quant_mb > 0 else float("inf")
    logger.info(
        "Taille : %.2f Mo (FP32) -> %.2f Mo (INT8)  | compression %.2fx",
        orig_mb, quant_mb, ratio,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantification INT8 de MobileNetV2 ONNX")
    parser.add_argument(
        "--mode",
        choices=["dynamic", "static", "both"],
        default="dynamic",
        help="Mode de quantification (défaut: dynamic)",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("models/mobilenetv2_fp32.onnx"),
        help="Modèle ONNX FP32 d'entrée",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models"),
        help="Dossier de sortie pour les modèles quantifiés",
    )
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=None,
        help="Dossier d'images pour la calibration (requis pour static/both)",
    )
    parser.add_argument(
        "--calibration-size",
        type=int,
        default=512,
        help="Nombre d'images pour la calibration (défaut: 512)",
    )
    parser.add_argument(
        "--no-per-channel",
        action="store_true",
        help="Désactive la quantif per-channel (per-tensor à la place)",
    )
    parser.add_argument(
        "--quant-format",
        choices=["qdq", "qoperator"],
        default="qdq",
        help=(
            "Format de stockage de la quantif (défaut: qdq). "
            "qdq est recommandé pour MobileNet (qoperator crashe sur le "
            "re-quantize du bias en per-channel sur certains canaux)."
        ),
    )
    parser.add_argument(
        "--calibrate-method",
        choices=["minmax", "entropy", "percentile"],
        default="percentile",
        help=(
            "Méthode de calibration des activations (défaut: percentile). "
            "minmax sensible aux outliers, percentile écrête le bruit, "
            "entropy minimise la KL divergence."
        ),
    )
    parser.add_argument(
        "--calib-percentile",
        type=float,
        default=99.999,
        help="Percentile pour --calibrate-method=percentile (défaut: 99.999)",
    )
    parser.add_argument(
        "--no-weight-symmetric",
        action="store_true",
        help="Désactive la quantif symétrique des poids (asymétrique à la place)",
    )
    parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="Saute l'étape quant_pre_process (déconseillé)",
    )
    args = parser.parse_args()

    if not args.input.exists():
        parser.error(f"Modèle FP32 introuvable : {args.input}. Lance d'abord export_onnx.py.")
    if args.mode in ("static", "both") and args.calibration_dir is None:
        parser.error("--calibration-dir est requis pour --mode static / both")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Pre-processing (shape inference + optims) — sortie partagée entre les deux modes
    if args.skip_preprocess:
        preprocessed_path = args.input
        logger.warning("Pre-processing sauté (--skip-preprocess)")
    else:
        preprocessed_path = args.output_dir / "mobilenetv2_fp32_preprocessed.onnx"
        preprocess_for_quantization(args.input, preprocessed_path)

    # 2) Quantification(s)
    if args.mode in ("dynamic", "both"):
        dyn_path = args.output_dir / "mobilenetv2_int8_dynamic.onnx"
        quantize_dynamic_int8(preprocessed_path, dyn_path)
        log_size_comparison(args.input, dyn_path)

    if args.mode in ("static", "both"):
        stat_path = args.output_dir / "mobilenetv2_int8_static.onnx"
        # Mapping str CLI -> enums onnxruntime.quantization
        quant_format_map = {
            "qdq": QuantFormat.QDQ,
            "qoperator": QuantFormat.QOperator,
        }
        calib_method_map = {
            "minmax": CalibrationMethod.MinMax,
            "entropy": CalibrationMethod.Entropy,
            "percentile": CalibrationMethod.Percentile,
        }
        quantize_static_int8(
            input_path=preprocessed_path,
            output_path=stat_path,
            calibration_dir=args.calibration_dir,
            n_samples=args.calibration_size,
            per_channel=not args.no_per_channel,
            quant_format=quant_format_map[args.quant_format],
            calibrate_method=calib_method_map[args.calibrate_method],
            calib_percentile=args.calib_percentile,
            weight_symmetric=not args.no_weight_symmetric,
        )
        log_size_comparison(args.input, stat_path)


if __name__ == "__main__":
    main()