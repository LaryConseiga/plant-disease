"""
Augmentation des 34 images extraites du docx.
Chaque image originale génère ~15 variantes → ~500 images total.
Utilise uniquement Pillow + numpy (pas de dépendances supplémentaires).
"""

import os
import random
import numpy as np
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

random.seed(42)
np.random.seed(42)

# ── Chemins ───────────────────────────────────────────────────────────────────
BASE_DIR   = Path(r"c:\Users\Larissa CONSEIGA\agrolaafi\plant-disease")
IMG_DIR    = BASE_DIR / "data" / "images"
AUG_DIR    = BASE_DIR / "data" / "images_augmented"

TARGET_SIZE = (224, 224)   # taille standard MobileNetV2


# ── Transformations ───────────────────────────────────────────────────────────

def resize(img):
    return img.resize(TARGET_SIZE, Image.LANCZOS)

def flip_horizontal(img):
    return ImageOps.mirror(img)

def flip_vertical(img):
    return ImageOps.flip(img)

def rotate(img, angle):
    return img.rotate(angle, expand=False, fillcolor=(0, 0, 0))

def brightness(img, factor):
    return ImageEnhance.Brightness(img).enhance(factor)

def contrast(img, factor):
    return ImageEnhance.Contrast(img).enhance(factor)

def saturation(img, factor):
    return ImageEnhance.Color(img).enhance(factor)

def sharpness(img, factor):
    return ImageEnhance.Sharpness(img).enhance(factor)

def blur(img, radius=1.5):
    return img.filter(ImageFilter.GaussianBlur(radius=radius))

def crop_zoom(img, factor=0.85):
    """Recadrage centré puis redimensionnement — simule un zoom."""
    w, h = img.size
    new_w, new_h = int(w * factor), int(h * factor)
    left   = (w - new_w) // 2
    top    = (h - new_h) // 2
    right  = left + new_w
    bottom = top  + new_h
    return img.crop((left, top, right, bottom)).resize((w, h), Image.LANCZOS)

def add_noise(img, strength=15):
    arr  = np.array(img).astype(np.int16)
    noise = np.random.randint(-strength, strength, arr.shape, dtype=np.int16)
    arr  = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def hue_shift(img, shift=20):
    """Décale la teinte de l'image HSV."""
    arr = np.array(img.convert("RGB")).astype(np.float32) / 255.0
    # Conversion RGB → HSV simplifiée via numpy
    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    mx  = np.max(arr, axis=2)
    mn  = np.min(arr, axis=2)
    df  = mx - mn + 1e-8
    h   = np.where(mx == r, (g - b) / df % 6,
          np.where(mx == g, (b - r) / df + 2,
                             (r - g) / df + 4)) / 6.0
    h   = (h + shift / 360.0) % 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(mx == 0, 0, df / mx)
    v   = mx
    # HSV → RGB
    i   = (h * 6).astype(int)
    f   = h * 6 - i
    p, q, t_ = v*(1-s), v*(1-f*s), v*(1-(1-f)*s)
    ri  = i % 6
    r2  = np.choose(ri, [v, q, p, p, t_, v])
    g2  = np.choose(ri, [t_, v, v, q, p, p])
    b2  = np.choose(ri, [p, p, t_, v, v, q])
    rgb = np.stack([r2, g2, b2], axis=2)
    return Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))


# ── Pipeline d'augmentation ───────────────────────────────────────────────────

def augment_pipeline(img):
    """
    Retourne une liste de (suffixe, image_augmentée).
    15 variantes par image originale.
    """
    variants = []

    # 1. Flip horizontal
    variants.append(("flip_h",   flip_horizontal(img)))
    # 2. Flip vertical
    variants.append(("flip_v",   flip_vertical(img)))
    # 3-4. Rotations
    variants.append(("rot_15",   rotate(img,  15)))
    variants.append(("rot_m15",  rotate(img, -15)))
    variants.append(("rot_30",   rotate(img,  30)))
    variants.append(("rot_m30",  rotate(img, -30)))
    # 5-6. Luminosité
    variants.append(("bright_up",  brightness(img, 1.3)))
    variants.append(("bright_dn",  brightness(img, 0.7)))
    # 7-8. Contraste
    variants.append(("contrast_up", contrast(img, 1.4)))
    variants.append(("contrast_dn", contrast(img, 0.7)))
    # 9. Saturation
    variants.append(("sat_up",   saturation(img, 1.5)))
    # 10. Flou
    variants.append(("blur",     blur(img, 1.5)))
    # 11. Zoom
    variants.append(("zoom",     crop_zoom(img, 0.82)))
    # 12. Bruit
    variants.append(("noise",    add_noise(img, 20)))
    # 13. Décalage de teinte
    variants.append(("hue",      hue_shift(img, 25)))

    return variants


# ── Traitement ────────────────────────────────────────────────────────────────

def process_disease_folder(disease_folder: Path, out_base: Path):
    disease_name = disease_folder.name
    out_dir      = out_base / disease_name
    out_dir.mkdir(parents=True, exist_ok=True)

    images = list(disease_folder.glob("*.jpeg")) + list(disease_folder.glob("*.png")) \
           + list(disease_folder.glob("*.jpg"))

    total = 0
    for img_path in images:
        img = Image.open(img_path).convert("RGB")
        img = resize(img)

        # Copie de l'original redimensionné
        stem = img_path.stem
        img.save(out_dir / f"{stem}_orig.jpg", "JPEG", quality=90)
        total += 1

        # Variantes augmentées
        for suffix, aug_img in augment_pipeline(img):
            out_name = f"{stem}_{suffix}.jpg"
            aug_img.save(out_dir / out_name, "JPEG", quality=90)
            total += 1

    return total


def main():
    print("=== Augmentation des images ===\n")

    if not IMG_DIR.exists():
        print(f"ERREUR: dossier images introuvable: {IMG_DIR}")
        return

    disease_folders = [f for f in IMG_DIR.iterdir() if f.is_dir()]
    if not disease_folders:
        print("Aucun dossier de maladie trouvé.")
        return

    grand_total = 0
    for folder in sorted(disease_folders):
        count = process_disease_folder(folder, AUG_DIR)
        print(f"  {folder.name:<40} {count:>3} images")
        grand_total += count

    print(f"\nTotal images augmentées : {grand_total}")
    print(f"Dossier de sortie       : {AUG_DIR}")


if __name__ == "__main__":
    main()
