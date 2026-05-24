"""
Utilitaires Imagenette : mapping wnid → indices ImageNet1K, parsing de structure,
preprocessing standard ImageNet (Resize 256 + CenterCrop 224 + normalisation).

Imagenette est un sous-ensemble de 10 classes ImageNet1K. Les classes sont
identifiées par leur wnid (WordNet ID) au format n01440764, qui est aussi
le nom de dossier dans imagenette2/{train,val}/.

Pour évaluer Top-1 sur un modèle entraîné sur ImageNet1K complet (1000 classes),
il faut savoir quel index ImageNet correspond à chaque classe Imagenette.

Le mapping ci-dessous est aligné sur l'ordre alphabétique des wnid utilisé
par torchvision (et donc par les poids MobileNet_V2_Weights.IMAGENET1K_V2).

Référence : https://github.com/fastai/imagenette
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
from PIL import Image


# ImageNet normalization standard (mêmes valeurs que torchvision.transforms)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# wnid -> (imagenet1k_index, label lisible)
# Indices vérifiés contre torchvision/imagenet_classes.txt (ordre lexicographique des wnid)
IMAGENETTE_CLASSES: dict[str, tuple[int, str]] = {
    "n01440764": (  0, "tench"),
    "n02102040": (217, "English springer"),
    "n02979186": (482, "cassette player"),
    "n03000684": (491, "chain saw"),
    "n03028079": (497, "church"),
    "n03394916": (566, "French horn"),
    "n03417042": (569, "garbage truck"),
    "n03425413": (571, "gas pump"),
    "n03445777": (574, "golf ball"),
    "n03888257": (701, "parachute"),
}


# Vues dérivées, prêtes à l'emploi
WNID_TO_IMAGENET_IDX: dict[str, int] = {
    wnid: idx for wnid, (idx, _) in IMAGENETTE_CLASSES.items()
}

IMAGENET_IDX_TO_WNID: dict[int, str] = {
    idx: wnid for wnid, idx in WNID_TO_IMAGENET_IDX.items()
}

# Les 10 indices ImageNet correspondant aux classes Imagenette
# Utile pour le "subset masking" : restreindre l'argmax à ces 10 classes.
IMAGENETTE_IMAGENET_INDICES: list[int] = sorted(WNID_TO_IMAGENET_IDX.values())


def preprocess_image(img_path: Path) -> np.ndarray:
    """Preprocessing ImageNet standard équivalent à torchvision.transforms.

    Pipeline strictement équivalent à :
        T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    Retourne (3, 224, 224) float32 en layout NCHW (sans la dim batch).

    NB : cette fonction est aussi utilisée par models/quantize.py pour la
    calibration, mais dupliquée là-bas pour éviter une dépendance circulaire
    entre les modules `models/` et `benchmark/`. À mutualiser proprement si
    le projet évolue vers un package Python installable.
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


def list_images_with_labels(split_dir: Path) -> list[tuple[Path, int, str]]:
    """Itère sur les images d'un split Imagenette et retourne leurs labels ImageNet.

    Args:
        split_dir : dossier train/ ou val/ de imagenette2

    Returns:
        Liste de tuples (chemin_image, imagenet_index, wnid)

    Lève FileNotFoundError si le dossier ou ses classes manquent.
    Ignore silencieusement les sous-dossiers dont le wnid n'est pas dans Imagenette.
    """
    split_dir = Path(split_dir)
    if not split_dir.exists():
        raise FileNotFoundError(f"Split introuvable : {split_dir}")

    results: list[tuple[Path, int, str]] = []
    for class_dir in sorted(split_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        wnid = class_dir.name
        if wnid not in WNID_TO_IMAGENET_IDX:
            continue  # ignore les dossiers parasites
        imagenet_idx = WNID_TO_IMAGENET_IDX[wnid]
        for img_path in sorted(class_dir.glob("*.JPEG")):
            results.append((img_path, imagenet_idx, wnid))

    return results


if __name__ == "__main__":
    # Petit auto-test
    print("Imagenette → ImageNet1K mapping :")
    print(f"{'wnid':<12} {'idx':>5}  label")
    print("-" * 40)
    for wnid, (idx, label) in IMAGENETTE_CLASSES.items():
        print(f"{wnid:<12} {idx:>5}  {label}")
    print(f"\nIndices ImageNet utilisés (triés) : {IMAGENETTE_IMAGENET_INDICES}")