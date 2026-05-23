"""
Étape 1 du pipeline — Export MobileNetV2 (pretrained ImageNet) en ONNX FP32.

Décisions de design :
- opset=17 : stable, supporté par onnxruntime>=1.16, suffisant pour Conv/BN/ReLU6
  qui composent MobileNetV2.
- dynamo=False : on force l'exporter TorchScript legacy. Le nouvel exporter
  dynamo (défaut depuis torch>=2.6) déprécie dynamic_axes et tente une
  conversion d'opset qui peut échouer silencieusement sur certains ops.
  Le graphe TorchScript reste plus prévisible et joue mieux avec
  onnxruntime.quantization.
- Axe batch dynamique uniquement : la spatial size reste fixée à 224x224
  (taille d'entrée standard ImageNet). Permet de batcher l'évaluation accuracy
  sans réexporter.
- do_constant_folding=True : pré-calcule les constantes (BN folding implicite
  côté exporter), réduit la taille et améliore les chances de bonne quantif.
- Poids MobileNet_V2_Weights.IMAGENET1K_V2 : meilleurs scores top-1 que V1
  (~72.15% vs ~71.88% sur ImageNet val), même architecture exportable.

Usage :
    python models/export_onnx.py
    python models/export_onnx.py --output models/mobilenetv2_fp32.onnx --opset 17
    python models/export_onnx.py --no-verify  # skip onnx.checker
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
import torchvision.models as models
import onnx


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def export_mobilenetv2(
    output_path: Path,
    opset: int = 17,
    verify: bool = True,
    seed: int = 0,
) -> None:
    """Charge MobileNetV2 pretrained ImageNet et exporte en ONNX FP32."""
    torch.manual_seed(seed)  # reproductibilité du dummy input

    logger.info("Chargement de MobileNetV2 (poids IMAGENET1K_V2)...")
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V2
    model = models.mobilenet_v2(weights=weights)
    model.eval()

    # Sanity check rapide en FP32 PyTorch avant export
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)
        torch_output = model(dummy_input)
    logger.info(
        "Forward pass PyTorch OK : input %s -> output %s",
        tuple(dummy_input.shape),
        tuple(torch_output.shape),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Export ONNX (opset=%d, exporter=TorchScript legacy) -> %s", opset, output_path)
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        opset_version=opset,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch"},
            "logits": {0: "batch"},
        },
        do_constant_folding=True,
        dynamo=False,
    )

    if verify:
        logger.info("Vérification onnx.checker.check_model...")
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        # Inspection rapide du graphe
        n_nodes = len(onnx_model.graph.node)
        n_initializers = len(onnx_model.graph.initializer)
        logger.info(
            "Modèle ONNX valide : %d nodes, %d initializers",
            n_nodes,
            n_initializers,
        )

    size_mb = output_path.stat().st_size / (1024 ** 2)
    logger.info("Taille du modèle exporté : %.2f Mo", size_mb)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export MobileNetV2 (ImageNet) en ONNX FP32"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/mobilenetv2_fp32.onnx"),
        help="Chemin de sortie .onnx (défaut: models/mobilenetv2_fp32.onnx)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="Version d'opset ONNX (défaut: 17)",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Désactive la vérification onnx.checker",
    )
    args = parser.parse_args()

    export_mobilenetv2(
        output_path=args.output,
        opset=args.opset,
        verify=not args.no_verify,
    )


if __name__ == "__main__":
    main()