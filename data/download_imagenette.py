"""
Télécharge et extrait Imagenette dans data/.

Imagenette : 10 classes ImageNet "facilement classifiables", libre d'accès,
créé par fast.ai. Idéal pour itérer rapidement sur la quantification.

Variantes disponibles :
- full  : taille originale  (~1.5 Go, ~10 min de DL)
- 320   : shortest side 320 (~340 Mo, ~2 min)  ← défaut
- 160   : shortest side 160 (~95 Mo,  ~30 s)

Pour MobileNetV2 (input 224x224), la variante 320 suffit largement :
le preprocessing standard fait Resize(256) -> CenterCrop(224), donc on
n'utilise jamais plus que 256px. La full size n'apporte rien pour la
calibration ou l'évaluation Top-1.

Usage :
    python data/download_imagenette.py                  # variante 320, défaut
    python data/download_imagenette.py --variant full   # full size
    python data/download_imagenette.py --variant 160    # rapide pour debug
    python data/download_imagenette.py --force          # force re-download
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import sys
import tarfile
import urllib.request
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


VARIANTS = {
    "full": {
        "url": "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz",
        "dirname": "imagenette2",
        "approx_size_mb": 1500,
    },
    "320": {
        "url": "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz",
        "dirname": "imagenette2-320",
        "approx_size_mb": 340,
    },
    "160": {
        "url": "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz",
        "dirname": "imagenette2-160",
        "approx_size_mb": 95,
    },
}


def _download_with_progress(url: str, dest: Path) -> None:
    """Télécharge avec progress bar minimaliste (pas de dépendance tqdm requise)."""
    logger.info("Téléchargement : %s -> %s", url, dest)

    def _hook(block_num: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return
        downloaded = block_num * block_size
        pct = min(100.0, 100.0 * downloaded / total_size)
        mb_done = downloaded / (1024 ** 2)
        mb_total = total_size / (1024 ** 2)
        bar = "#" * int(pct / 2) + "-" * (50 - int(pct / 2))
        sys.stdout.write(f"\r  [{bar}] {pct:5.1f}%  {mb_done:7.1f} / {mb_total:7.1f} Mo")
        sys.stdout.flush()

    urllib.request.urlretrieve(url, dest, reporthook=_hook)
    sys.stdout.write("\n")
    sys.stdout.flush()


def _extract_tgz(archive: Path, dest_dir: Path) -> None:
    """Extrait un .tgz dans dest_dir. Sûreté : refuse les chemins absolus / traversal."""
    logger.info("Extraction : %s -> %s", archive, dest_dir)
    with tarfile.open(archive, "r:gz") as tar:
        # Filtre de sécurité (Python 3.12+ requiert un filter explicite)
        def safe_filter(member, path):
            name = Path(member.name)
            if name.is_absolute() or ".." in name.parts:
                raise ValueError(f"Chemin suspect dans l'archive : {member.name}")
            return member
        tar.extractall(path=dest_dir, filter=safe_filter)


def _check_structure(extracted_dir: Path) -> tuple[int, int]:
    """Vérifie la structure imagenette train/val et retourne (n_train, n_val)."""
    train = extracted_dir / "train"
    val = extracted_dir / "val"
    if not train.exists() or not val.exists():
        raise RuntimeError(
            f"Structure inattendue dans {extracted_dir} : "
            f"train/ ou val/ manquant"
        )

    n_train = sum(1 for _ in train.rglob("*.JPEG"))
    n_val = sum(1 for _ in val.rglob("*.JPEG"))
    n_classes_train = sum(1 for p in train.iterdir() if p.is_dir())
    n_classes_val = sum(1 for p in val.iterdir() if p.is_dir())

    logger.info("Structure validée :")
    logger.info("  train : %d classes, %d images", n_classes_train, n_train)
    logger.info("  val   : %d classes, %d images", n_classes_val, n_val)

    if n_classes_train != 10 or n_classes_val != 10:
        logger.warning("Attendu 10 classes par split (Imagenette), trouvé "
                       f"train={n_classes_train}, val={n_classes_val}")

    return n_train, n_val


def main() -> None:
    parser = argparse.ArgumentParser(description="Télécharge Imagenette dans data/")
    parser.add_argument(
        "--variant",
        choices=list(VARIANTS.keys()),
        default="320",
        help="Variante à télécharger (défaut: 320)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Dossier racine pour les données (défaut: data/)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force le re-téléchargement même si l'archive existe déjà",
    )
    parser.add_argument(
        "--keep-archive",
        action="store_true",
        help="Garde le .tgz après extraction (sinon supprimé pour gagner de la place)",
    )
    args = parser.parse_args()

    info = VARIANTS[args.variant]
    args.data_dir.mkdir(parents=True, exist_ok=True)

    archive_path = args.data_dir / Path(info["url"]).name
    extracted_path = args.data_dir / info["dirname"]

    # Idempotence : si déjà extrait, on ne refait rien
    if extracted_path.exists() and not args.force:
        logger.info("Dossier %s déjà présent, vérification de la structure...",
                    extracted_path)
        _check_structure(extracted_path)
        logger.info("Tout est en place. Pour forcer le re-téléchargement : --force")
        return

    # Téléchargement
    if archive_path.exists() and not args.force:
        logger.info("Archive %s déjà présente (taille %.1f Mo), skip du DL",
                    archive_path, archive_path.stat().st_size / 1024**2)
    else:
        logger.info("Variante: %s (~%d Mo)", args.variant, info["approx_size_mb"])
        _download_with_progress(info["url"], archive_path)

    # Extraction
    _extract_tgz(archive_path, args.data_dir)
    _check_structure(extracted_path)

    # Nettoyage
    if not args.keep_archive:
        logger.info("Suppression de l'archive %s (utiliser --keep-archive pour conserver)",
                    archive_path)
        archive_path.unlink()

    logger.info("Prêt. Pour la calibration :")
    logger.info("  uv run models/quantize.py --mode static \\")
    logger.info("    --calibration-dir %s/train \\", extracted_path)
    logger.info("    --calibration-size 512")


if __name__ == "__main__":
    main()