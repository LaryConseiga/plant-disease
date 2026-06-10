"""
Entraînement du modèle de détection de maladies des plantes — AgroLaafi
Architecture  : MobileNetV2 (transfer learning depuis ImageNet)
Cible hardware: Laafi Box (ESP32-S3 / Raspberry Pi Zero 2W) via TFLite
"""

import os
import json
import random
import shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")          # pas de fenêtre graphique requise
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # cache les warnings TF inutiles
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)

print(f"TensorFlow {tf.__version__}  |  GPU: {tf.config.list_physical_devices('GPU')}")

# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR   = Path(r"c:\Users\Larissa CONSEIGA\agrolaafi\plant-disease")
DATA_DIR   = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
PLOTS_DIR  = BASE_DIR / "plots"

MODELS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

IMG_SIZE    = (224, 224)       # taille standard MobileNetV2
BATCH_SIZE  = 16               # réduit pour éviter les OOM sur petites machines
EPOCHS_TOP  = 15               # Phase 1 : top layers seulement
EPOCHS_FINE = 25               # Phase 2 : fine-tuning couches profondes
SEED        = 42
VAL_SPLIT   = 0.15             # 15% validation
TEST_SPLIT  = 0.10             # 10% test
MIN_IMAGES  = 10               # classes avec moins de N images → ignorées

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ─────────────────────────────────────────────────────────────────────────────
# 2. MAPPING DOSSIERS → LABEL CANONIQUE
#    Unifie les noms issus de 3 sources : docx, iNaturalist, Wikimedia
# ─────────────────────────────────────────────────────────────────────────────

LABEL_MAP = {
    # ── Source docx (images_augmented) ──────────────────────────────────────
    "acariose_bronzée":          "Acariose bronzee",
    "alternariose":               "Alternariose",
    "anthracnose":                "Anthracnose",
    "araignée_rouge_tétranyque": "Araignee rouge",
    "botrytis":                   "Botrytis",
    "carence_en_azote":          "Carence azote",
    "carence_en_potassium":      "Carence potassium",
    "cmv":                        "CMV",
    "coup_de_soleil":            "Coup de soleil",
    "flétrissement_bactérien":   "Fletrissement bacterien",
    "fonte_de_semis":            "Fonte de semis",
    "fusariose":                  "Fusariose",
    "gale_bactérienne":          "Gale bacterienne",
    "mildiou":                    "Mildiou",
    "mineuse_de_la_tomate":     "Mineuse tomate",
    "mouche_blanche":            "Mouche blanche",
    "nécrose_apicale":          "Necrose apicale",
    "nématodes_à_galles":       "Nematodes galles",
    "noctuelle_de_la_tomate":   "Noctuelle tomate",
    "oidium":                     "Oidium",
    "puceron":                    "Puceron",
    "stemphyliose":               "Stemphyliose",
    "tarsonème":                  "Tarsoneme",
    "thrips":                     "Thrips",
    "tmv":                        "TMV",
    "tylcv":                      "TYLCV",
    "verticilliose":              "Verticilliose",
    # ── Source iNaturalist / Wikimedia (dossiers scraped) ───────────────────
    "acariose_bronzee":          "Acariose bronzee",
    "araignee_rouge":            "Araignee rouge",
    "fletrissement_bacterien":   "Fletrissement bacterien",
    "gale_bacterienne":          "Gale bacterienne",
    "necrose_apicale":           "Necrose apicale",
    "nematodes":                  "Nematodes galles",
    "tarsoneme":                  "Tarsoneme",
    "mineuse_de_la_tomate":     "Mineuse tomate",
    "mouche_blanche":            "Mouche blanche",
    "noctuelle_de_la_tomate":   "Noctuelle tomate",
}


def get_label(folder_name: str) -> str:
    """Retourne le label canonique d'un dossier, ou le dossier lui-même."""
    return LABEL_MAP.get(folder_name, folder_name.replace("_", " ").title())


# ─────────────────────────────────────────────────────────────────────────────
# 3. COLLECTE DES IMAGES (toutes sources confondues)
# ─────────────────────────────────────────────────────────────────────────────

SOURCES = [
    DATA_DIR / "images_augmented",                   # docx + augmentation
    DATA_DIR / "scraped" / "inaturalist",            # iNaturalist API
    DATA_DIR / "scraped" / "wikimedia",              # Wikimedia Commons
]

def collect_all_images() -> dict[str, list[Path]]:
    """
    Scanne toutes les sources et retourne un dict {label: [chemins images]}.
    """
    label_to_paths: dict[str, list[Path]] = {}

    for source_dir in SOURCES:
        if not source_dir.exists():
            print(f"  [SKIP] Source introuvable : {source_dir.name}")
            continue
        for folder in sorted(source_dir.iterdir()):
            if not folder.is_dir():
                continue
            label  = get_label(folder.name)
            images = [
                p for p in folder.iterdir()
                if p.suffix.lower() in (".jpg", ".jpeg", ".png")
            ]
            if not images:
                continue
            label_to_paths.setdefault(label, []).extend(images)

    return label_to_paths


def split_dataset(
    label_to_paths: dict[str, list[Path]]
) -> tuple[list, list, list, list[str]]:
    """
    Divise les images en train / val / test.
    Retourne (train_pairs, val_pairs, test_pairs, class_names)
    où chaque pair est (chemin_image, index_classe).
    """
    # Filtre les classes avec trop peu d'images
    filtered = {
        label: paths
        for label, paths in label_to_paths.items()
        if len(paths) >= MIN_IMAGES
    }

    class_names = sorted(filtered.keys())
    label_to_idx = {label: i for i, label in enumerate(class_names)}

    train, val, test = [], [], []

    for label, paths in filtered.items():
        random.shuffle(paths)
        idx   = label_to_idx[label]
        n     = len(paths)
        n_val  = max(1, int(n * VAL_SPLIT))
        n_test = max(1, int(n * TEST_SPLIT))

        test  += [(p, idx) for p in paths[:n_test]]
        val   += [(p, idx) for p in paths[n_test:n_test + n_val]]
        train += [(p, idx) for p in paths[n_test + n_val:]]

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    return train, val, test, class_names


# ─────────────────────────────────────────────────────────────────────────────
# 4. PIPELINE tf.data
# ─────────────────────────────────────────────────────────────────────────────

def parse_image(path: str, label: int, num_classes: int,
                augment: bool = False) -> tuple:
    """Lit, redimensionne et optionnellement augmente une image."""
    raw   = tf.io.read_file(path)
    image = tf.image.decode_image(raw, channels=3, expand_animations=False)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0   # normalisation [0, 1]

    if augment:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_contrast(image, 0.8, 1.2)
        image = tf.image.random_saturation(image, 0.8, 1.2)
        image = tf.clip_by_value(image, 0.0, 1.0)

    # MobileNetV2 attend des valeurs dans [-1, 1]
    image = image * 2.0 - 1.0

    label_oh = tf.one_hot(label, num_classes)
    return image, label_oh


def make_dataset(pairs: list, num_classes: int,
                 augment: bool = False) -> tf.data.Dataset:
    paths  = [str(p) for p, _ in pairs]
    labels = [l for _, l in pairs]

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(
        lambda p, l: parse_image(p, l, num_classes, augment),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    if augment:
        ds = ds.shuffle(buffer_size=min(len(pairs), 2000), seed=SEED)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


# ─────────────────────────────────────────────────────────────────────────────
# 5. MODÈLE MobileNetV2 + TÊTE DE CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def build_model(num_classes: int) -> Model:
    """
    MobileNetV2 pré-entraîné sur ImageNet + tête de classification custom.
    Les couches de base sont gelées pour la Phase 1.
    """
    base = MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False    # gelé en Phase 1

    inputs = keras.Input(shape=(*IMG_SIZE, 3), name="image")
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(256, activation="relu", name="dense1")(x)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Dropout(0.4, name="dropout1")(x)
    x = layers.Dense(128, activation="relu", name="dense2")(x)
    x = layers.Dropout(0.3, name="dropout2")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = Model(inputs, outputs, name="AgroLaafi_v1")
    return model, base


def unfreeze_top_layers(base_model: Model, num_layers: int = 30):
    """Dégèle les N dernières couches pour le fine-tuning (Phase 2)."""
    base_model.trainable = True
    for layer in base_model.layers[:-num_layers]:
        layer.trainable = False
    trainable_count = sum(1 for l in base_model.layers if l.trainable)
    print(f"  Couches base entraînables : {trainable_count}/{len(base_model.layers)}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. CALLBACKS
# ─────────────────────────────────────────────────────────────────────────────

def get_callbacks(phase: int, checkpoint_path: Path):
    return [
        ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_accuracy",
            patience=7 if phase == 1 else 10,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-7,
            verbose=1,
        ),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 7. VISUALISATIONS
# ─────────────────────────────────────────────────────────────────────────────

def plot_history(history1, history2, save_path: Path):
    """Courbes d'entraînement des 2 phases."""
    acc  = history1.history["accuracy"]      + history2.history["accuracy"]
    val  = history1.history["val_accuracy"]  + history2.history["val_accuracy"]
    loss = history1.history["loss"]          + history2.history["loss"]
    vloss= history1.history["val_loss"]      + history2.history["val_loss"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ep = range(1, len(acc) + 1)
    sep = len(history1.history["accuracy"])

    axes[0].plot(ep, acc,  label="Train Accuracy")
    axes[0].plot(ep, val,  label="Val Accuracy")
    axes[0].axvline(sep, color="gray", linestyle="--", label="Fine-tuning")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Époque")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(ep, loss,  label="Train Loss")
    axes[1].plot(ep, vloss, label="Val Loss")
    axes[1].axvline(sep, color="gray", linestyle="--", label="Fine-tuning")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Époque")
    axes[1].legend()
    axes[1].grid(True)

    plt.suptitle("AgroLaafi — Courbes d'entraînement MobileNetV2", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Courbes sauvegardées : {save_path.name}")


def plot_class_distribution(label_to_paths: dict, class_names: list,
                             save_path: Path):
    """Histogramme du nombre d'images par classe."""
    counts = [len(label_to_paths.get(c, [])) for c in class_names]
    fig, ax = plt.subplots(figsize=(16, 6))
    bars = ax.barh(class_names, counts, color="steelblue")
    ax.bar_label(bars, padding=3)
    ax.set_xlabel("Nombre d'images")
    ax.set_title("Distribution des images par classe")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Distribution sauvegardée : {save_path.name}")


def plot_confusion_matrix(y_true, y_pred, class_names: list, save_path: Path):
    """Matrice de confusion normalisée."""
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, normalize="true")

    fig, ax = plt.subplots(figsize=(18, 16))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.03)

    n = len(class_names)
    ax.set_xticks(range(n));  ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n));  ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("Prédit");  ax.set_ylabel("Réel")
    ax.set_title("Matrice de confusion (normalisée)", fontsize=13)

    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{cm[i,j]:.2f}", ha="center", va="center",
                    fontsize=6, color="white" if cm[i,j] > 0.5 else "black")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Matrice de confusion sauvegardée : {save_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# 8. EXPORT TFLite (pour le Laafi Box)
# ─────────────────────────────────────────────────────────────────────────────

def export_tflite(model: Model, class_names: list, out_dir: Path):
    """
    Convertit le modèle Keras en TFLite quantifié (int8/float16).
    Le fichier .tflite est directement utilisable sur ESP32-S3 ou RPi Zero 2W.
    """
    # Version float32 (la plus précise)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    path_f32 = out_dir / "agrolaafi_v1_float32.tflite"
    path_f32.write_bytes(tflite_model)
    print(f"  TFLite float32 : {path_f32.stat().st_size / 1024:.0f} KB")

    # Version float16 (2× plus légère, quasi-même précision)
    converter2 = tf.lite.TFLiteConverter.from_keras_model(model)
    converter2.optimizations = [tf.lite.Optimize.DEFAULT]
    converter2.target_spec.supported_types = [tf.float16]
    tflite_f16 = converter2.convert()
    path_f16 = out_dir / "agrolaafi_v1_float16.tflite"
    path_f16.write_bytes(tflite_f16)
    print(f"  TFLite float16 : {path_f16.stat().st_size / 1024:.0f} KB")

    # Sauvegarde des noms de classes (nécessaire pour l'inférence)
    labels_path = out_dir / "class_names.json"
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)
    print(f"  Labels JSON     : {labels_path.name}")

    return path_f16


# ─────────────────────────────────────────────────────────────────────────────
# 9. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print("   AgroLaafi — Entraînement MobileNetV2")
    print("="*60 + "\n")

    # ── Collecte des données ──────────────────────────────────────────────────
    print("[ 1/7 ] Collecte des images ...")
    label_to_paths = collect_all_images()

    total_images = sum(len(v) for v in label_to_paths.values())
    print(f"  {len(label_to_paths)} classes  |  {total_images} images total\n")
    for label, paths in sorted(label_to_paths.items()):
        status = "OK" if len(paths) >= MIN_IMAGES else "FAIBLE"
        print(f"  [{status}] {label:<35} {len(paths):>4} images")

    # ── Split train / val / test ──────────────────────────────────────────────
    print(f"\n[ 2/7 ] Split train/val/test (val={VAL_SPLIT:.0%}, test={TEST_SPLIT:.0%}) ...")
    train_pairs, val_pairs, test_pairs, class_names = split_dataset(label_to_paths)
    num_classes = len(class_names)

    print(f"  Classes : {num_classes}")
    print(f"  Train   : {len(train_pairs)} images")
    print(f"  Val     : {len(val_pairs)} images")
    print(f"  Test    : {len(test_pairs)} images")

    # Sauvegarde des noms de classes
    with open(MODELS_DIR / "class_names.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)

    # Distribution
    plot_class_distribution(
        label_to_paths, class_names,
        PLOTS_DIR / "class_distribution.png"
    )

    # ── Datasets tf.data ──────────────────────────────────────────────────────
    print("\n[ 3/7 ] Création des pipelines tf.data ...")
    train_ds = make_dataset(train_pairs, num_classes, augment=True)
    val_ds   = make_dataset(val_pairs,   num_classes, augment=False)
    test_ds  = make_dataset(test_pairs,  num_classes, augment=False)

    # ── Construction du modèle ────────────────────────────────────────────────
    print(f"\n[ 4/7 ] Construction du modèle MobileNetV2 ({num_classes} classes) ...")
    model, base_model = build_model(num_classes)
    model.summary(line_length=80)

    # ── Phase 1 : entraînement top layers ────────────────────────────────────
    print("\n[ 5/7 ] Phase 1 — Top layers (base gelée) ...")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_acc")],
    )

    ckpt1 = MODELS_DIR / "best_phase1.keras"
    history1 = model.fit(
        train_ds,
        epochs=EPOCHS_TOP,
        validation_data=val_ds,
        callbacks=get_callbacks(phase=1, checkpoint_path=ckpt1),
        verbose=1,
    )

    # ── Phase 2 : fine-tuning ─────────────────────────────────────────────────
    print("\n[ 6/7 ] Phase 2 — Fine-tuning (30 dernières couches) ...")
    unfreeze_top_layers(base_model, num_layers=30)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_acc")],
    )

    ckpt2 = MODELS_DIR / "best_phase2.keras"
    history2 = model.fit(
        train_ds,
        epochs=EPOCHS_FINE,
        validation_data=val_ds,
        callbacks=get_callbacks(phase=2, checkpoint_path=ckpt2),
        verbose=1,
    )

    # ── Évaluation finale ─────────────────────────────────────────────────────
    print("\n[ 7/7 ] Évaluation sur le jeu de test ...")
    # Charge le meilleur modèle Phase 2
    best_model = tf.keras.models.load_model(ckpt2)
    test_loss, test_acc, test_top3 = best_model.evaluate(test_ds, verbose=0)
    print(f"\n  Test Accuracy     : {test_acc:.4f}  ({test_acc*100:.1f}%)")
    print(f"  Test Top-3 Acc    : {test_top3:.4f}  ({test_top3*100:.1f}%)")
    print(f"  Test Loss         : {test_loss:.4f}")

    # Matrice de confusion
    y_true, y_pred = [], []
    for images, labels in test_ds:
        preds = best_model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    plot_confusion_matrix(y_true, y_pred, class_names,
                          PLOTS_DIR / "confusion_matrix.png")

    # Courbes d'entraînement
    plot_history(history1, history2, PLOTS_DIR / "training_curves.png")

    # ── Export TFLite ─────────────────────────────────────────────────────────
    print("\n[ Export ] Conversion TFLite pour Laafi Box ...")
    tflite_path = export_tflite(best_model, class_names, MODELS_DIR)

    # ── Résumé final ──────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("   ENTRAÎNEMENT TERMINÉ")
    print("="*60)
    print(f"  Accuracy test     : {test_acc*100:.1f}%")
    print(f"  Top-3 accuracy    : {test_top3*100:.1f}%")
    print(f"  Modèle sauvegardé : {ckpt2}")
    print(f"  TFLite Laafi Box  : {tflite_path}")
    print(f"  Plots             : {PLOTS_DIR}")

    # Résultats JSON
    results = {
        "test_accuracy":      round(float(test_acc),  4),
        "test_top3_accuracy": round(float(test_top3), 4),
        "test_loss":          round(float(test_loss),  4),
        "num_classes":        num_classes,
        "class_names":        class_names,
        "train_images":       len(train_pairs),
        "val_images":         len(val_pairs),
        "test_images":        len(test_pairs),
    }
    with open(MODELS_DIR / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"  Résultats JSON    : models/results.json\n")


if __name__ == "__main__":
    main()
