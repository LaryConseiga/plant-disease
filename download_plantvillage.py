"""
Téléchargement du dataset PlantVillage via tensorflow-datasets.
Extrait les classes pertinentes et les mappe vers la taxonomie AgroLaafi.
Sortie : data/scraped/plantvillage/<dossier_classe>/<image>.jpg
"""

import os
import numpy as np
from pathlib import Path
from collections import defaultdict

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import tensorflow_datasets as tfds

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR  = BASE_DIR / "data" / "scraped" / "plantvillage"

# Images max extraites par dossier de sortie (pas par classe PlantVillage —
# plusieurs classes PV peuvent pointer vers le même dossier)
MAX_PER_FOLDER = 300

# Mapping : label PlantVillage → (nom_dossier_sortie, label_canonique_AgroLaafi)
# Plusieurs classes PV peuvent partager le même dossier (même maladie, plante diff.)
PLANTVILLAGE_MAP = {
    "Tomato___Bacterial_spot":                        ("tomato_bacterial_spot",  "Gale bacterienne"),
    "Tomato___Early_blight":                          ("tomato_early_blight",    "Alternariose"),
    "Tomato___Late_blight":                           ("tomato_late_blight",     "Mildiou"),
    "Tomato___Leaf_Mold":                             ("tomato_leaf_mold",       "Mildiou"),
    "Tomato___Septoria_leaf_spot":                    ("tomato_septoria",        "Stemphyliose"),
    "Tomato___Spider_mites Two-spotted_spider_mite":  ("tomato_spider_mites",    "Araignee rouge"),
    "Tomato___Target_Spot":                           ("tomato_target_spot",     "Anthracnose"),
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus":         ("tomato_tylcv",           "TYLCV"),
    "Tomato___Tomato_mosaic_virus":                   ("tomato_mosaic_virus",    "TMV"),
    "Potato___Early_blight":                          ("potato_early_blight",    "Alternariose"),
    "Potato___Late_blight":                           ("potato_late_blight",     "Mildiou"),
    "Pepper,_bell___Bacterial_spot":                  ("pepper_bacterial_spot",  "Gale bacterienne"),
    "Squash___Powdery_mildew":                        ("squash_powdery_mildew",  "Oidium"),
    "Tomato___healthy":                               ("tomato_healthy",         "Saine"),
}

# Noms de dossiers → label canonique (pour le LABEL_MAP de train_model.py)
FOLDER_TO_LABEL = {folder: label for folder, label in PLANTVILLAGE_MAP.values()}


def save_image(image_tensor: tf.Tensor, out_path: Path) -> bool:
    """Encode un tenseur image en JPEG et l'écrit sur disque."""
    try:
        img_np = image_tensor.numpy()
        encoded = tf.image.encode_jpeg(img_np, quality=92)
        out_path.write_bytes(encoded.numpy())
        return True
    except Exception as e:
        print(f"    [ERR] {out_path.name}: {e}")
        return False


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  AgroLaafi — Extraction PlantVillage")
    print("=" * 60)
    print(f"  Sortie      : {OUT_DIR}")
    print(f"  Max/dossier : {MAX_PER_FOLDER} images")
    print()

    # ── Chargement du dataset ─────────────────────────────────────────────────
    print("[ 1/3 ] Chargement des métadonnées PlantVillage ...")
    ds_info = tfds.builder("plant_village").info
    label_names = ds_info.features["label"].names
    print(f"  {len(label_names)} classes disponibles dans PlantVillage")

    # Vérifie que nos classes cibles existent bien dans le dataset
    missing = [k for k in PLANTVILLAGE_MAP if k not in label_names]
    if missing:
        print(f"  [WARN] Classes introuvables dans PlantVillage : {missing}")

    label_to_idx = {name: i for i, name in enumerate(label_names)}
    target_indices = {
        label_to_idx[pv_label]: (folder, canon_label)
        for pv_label, (folder, canon_label) in PLANTVILLAGE_MAP.items()
        if pv_label in label_to_idx
    }

    # ── Création des dossiers de sortie ───────────────────────────────────────
    for folder, _ in FOLDER_TO_LABEL.items():
        (OUT_DIR / folder).mkdir(parents=True, exist_ok=True)

    # ── Compteurs d'images existantes ─────────────────────────────────────────
    folder_counts = defaultdict(int)
    for folder in FOLDER_TO_LABEL:
        folder_counts[folder] = len(list((OUT_DIR / folder).glob("*.jpg")))
        if folder_counts[folder] > 0:
            print(f"  [EXIST] {folder}: {folder_counts[folder]} images déjà présentes")

    if all(folder_counts[f] >= MAX_PER_FOLDER for f in FOLDER_TO_LABEL):
        print("\n  Tous les dossiers sont déjà complets. Rien à faire.")
        _print_summary(folder_counts)
        return

    # ── Téléchargement et extraction ──────────────────────────────────────────
    print("\n[ 2/3 ] Téléchargement et extraction des images ...")
    print("  (Première exécution : ~1.5 GB à télécharger, quelques minutes)")
    print()

    ds = tfds.load(
        "plant_village",
        split="train",
        shuffle_files=False,
        as_supervised=False,
    )

    total_saved = 0
    for example in ds:
        label_idx = int(example["label"].numpy())
        if label_idx not in target_indices:
            continue

        folder, _ = target_indices[label_idx]
        if folder_counts[folder] >= MAX_PER_FOLDER:
            continue

        out_path = OUT_DIR / folder / f"pv_{folder_counts[folder]:04d}.jpg"
        if save_image(example["image"], out_path):
            folder_counts[folder] += 1
            total_saved += 1

        # Affichage de progression tous les 100 enregistrements
        if total_saved % 100 == 0 and total_saved > 0:
            print(f"  {total_saved} images extraites ...")

        # Arrêt anticipé si tous les dossiers sont pleins
        if all(folder_counts[f] >= MAX_PER_FOLDER for f in FOLDER_TO_LABEL):
            break

    # ── Résumé ────────────────────────────────────────────────────────────────
    print(f"\n[ 3/3 ] {total_saved} nouvelles images extraites")
    _print_summary(folder_counts)


def _print_summary(folder_counts: dict):
    print("\n  Résumé par dossier :")
    print(f"  {'Dossier':<35} {'Label AgroLaafi':<25} {'Images':>6}")
    print("  " + "-" * 68)
    grand_total = 0
    for folder, canon_label in sorted(FOLDER_TO_LABEL.items()):
        n = folder_counts[folder]
        grand_total += n
        status = "OK" if n >= MAX_PER_FOLDER else f"{n}"
        print(f"  {folder:<35} {canon_label:<25} {status:>6}")
    print("  " + "-" * 68)
    print(f"  {'TOTAL':<62} {grand_total:>6}")
    print()
    print("  Prochaine étape : python3 train_model.py")


if __name__ == "__main__":
    main()
