"""
Extraction des images du fichier "Maladie word.docx"
- Associe chaque image à la maladie correspondante
- Crée la structure de dossiers data/images/{maladie}/
- Génère data/image_labels.csv pour l'entraînement du modèle
"""

import os
import re
import csv
import zipfile
import shutil
from pathlib import Path
from lxml import etree

# ── Chemins ──────────────────────────────────────────────────────────────────
BASE_DIR   = Path(r"c:\Users\Larissa CONSEIGA\agrolaafi\plant-disease")
DOCX_PATH  = BASE_DIR / "Maladie word.docx"
IMG_DIR    = BASE_DIR / "data" / "images"
CSV_OUT    = BASE_DIR / "data" / "image_labels.csv"

# Namespaces Word
NS = {
    "w":   "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "r":   "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "a":   "http://schemas.openxmlformats.org/drawingml/2006/main",
    "pic": "http://schemas.openxmlformats.org/drawingml/2006/picture",
    "wp":  "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def slugify(name: str) -> str:
    """Transforme un nom de maladie en nom de dossier valide."""
    name = name.lower().strip()
    name = re.sub(r"[^\w\s-]", "", name)
    name = re.sub(r"[\s]+", "_", name)
    return name


# Premiers mots qui indiquent que le paragraphe est une description, pas un titre de maladie
_NOT_TITLE_STARTS = {
    "lutte", "symptomes", "symptômes", "deficit", "déficit",
    "champignon", "bacterie", "bactérie", "acarien", "acariens",
    "lepidoptere", "lépidoptère", "insectes", "maladie",
    "oomycete", "oomycète", "ascomycete", "ascomycète",
    "tobamovirus", "begomovirus",
    "varietes", "variétés", "semences", "rotation",
}

# Normalisation des labels détectés → nom canonique de la maladie
_LABEL_ALIASES = {
    "Maladie cryptogamique": "Fonte de semis",
    "Puceron Insecte suceur": "Puceron",
}


def extract_disease_name(text: str) -> str | None:
    """
    Détecte si un paragraphe est un titre de maladie.
    Les titres contiennent le nom de maladie + nom scientifique entre parenthèses,
    ou commencent par un bullet '•', ou se terminent par ':'.
    Les paragraphes de traitement/symptômes sont exclus via _NOT_TITLE_STARTS.
    """
    text = text.strip()
    if not text:
        return None

    # Vérifie que ça ne commence pas par un mot de description
    first_word = text.lstrip("•· ").split()[0].lower().rstrip(":") if text.split() else ""
    if first_word in _NOT_TITLE_STARTS:
        return None

    # Retire le bullet '•' si présent
    clean = re.sub(r"^[•·]\s*", "", text)

    # Patterns de titres — ordre du plus spécifique au moins spécifique
    patterns = [
        # "Nom (Scientifique) ..." ou "Nom (Scientifique) – desc"  (espace optionnel après "(")
        r"^([A-ZÀ-Ö][A-Za-zÀ-öø-ÿ /\-]{2,40})\s*\(\s*[A-Z]",
        # "Nom :" ou "Nom : desc courte"
        r"^([A-ZÀ-Ö][A-Za-zÀ-öø-ÿ /\-]{2,40})\s*:",
        # "Nom – desc" (tiret long)
        r"^([A-ZÀ-Ö][A-Za-zÀ-öø-ÿ /\-]{2,40})\s*[–—]",
        # Titre seul (uniquement si <= 5 mots)
        r"^([A-ZÀ-Ö][A-Za-zÀ-öø-ÿ /\-]{2,40})\s*$",
    ]
    for pat in patterns:
        m = re.match(pat, clean)
        if m:
            candidate = m.group(1).strip()
            if len(candidate.split()) <= 6:
                return candidate
    return None


def get_rids_from_paragraph(para_xml) -> list[str]:
    """Retourne la liste des rId d'images dans un paragraphe XML."""
    rids = []
    # Inline images
    for blip in para_xml.findall(".//a:blip", NS):
        rid = blip.get(f"{{{NS['r']}}}embed")
        if rid:
            rids.append(rid)
    # Anchored images
    for blip in para_xml.findall(".//wp:anchor//a:blip", NS):
        rid = blip.get(f"{{{NS['r']}}}embed")
        if rid:
            rids.append(rid)
    return rids


# ── Lecture du document XML ───────────────────────────────────────────────────

def parse_document(docx_path: Path):
    """
    Retourne:
      - paragraphs : liste de (text, [rids])
      - rels       : dict rId -> fichier média dans le ZIP
    """
    with zipfile.ZipFile(docx_path) as z:
        # Contenu principal
        doc_xml  = z.read("word/document.xml")
        rels_xml = z.read("word/_rels/document.xml.rels")
        media_files = {
            name.split("/")[-1]: name
            for name in z.namelist()
            if name.startswith("word/media/")
        }

    # Relations rId -> target (nom de fichier)
    rels_tree = etree.fromstring(rels_xml)
    rels = {}
    for rel in rels_tree:
        rid    = rel.get("Id")
        target = rel.get("Target", "")
        if "media" in target:
            filename = target.split("/")[-1]
            rels[rid] = filename

    # Paragraphes
    doc_tree   = etree.fromstring(doc_xml)
    body       = doc_tree.find(".//w:body", NS)
    paragraphs = []
    for para in body.findall("w:p", NS):
        # Texte brut
        texts = para.findall(".//w:t", NS)
        text  = "".join(t.text or "" for t in texts).strip()
        rids  = get_rids_from_paragraph(para)
        paragraphs.append((text, rids))

    return paragraphs, rels, media_files


# ── Association image → maladie ───────────────────────────────────────────────

def build_label_map(paragraphs, rels, media_files):
    """
    Parcourt les paragraphes dans l'ordre et associe chaque rId
    à la dernière maladie détectée.
    Retourne liste de (rid, disease_name, media_filename).
    """
    current_disease = None
    labeled = []

    for text, rids in paragraphs:
        # Tente de détecter un nouveau titre de maladie
        detected = extract_disease_name(text)
        if detected:
            current_disease = detected

        if rids and current_disease:
            canonical = _LABEL_ALIASES.get(current_disease, current_disease)
            for rid in rids:
                media_file = rels.get(rid)
                if media_file and media_file in media_files:
                    labeled.append((rid, canonical, media_file))

    return labeled


# ── Extraction et copie des images ────────────────────────────────────────────

def extract_images(docx_path: Path, labeled, media_files, img_dir: Path):
    """
    Extrait les images du ZIP vers data/images/{maladie}/img_XXX.{ext}
    Retourne liste de dicts pour le CSV.
    """
    records = []
    counters = {}

    with zipfile.ZipFile(docx_path) as z:
        for rid, disease, media_file in labeled:
            folder_name = slugify(disease)
            out_dir     = img_dir / folder_name
            out_dir.mkdir(parents=True, exist_ok=True)

            counters[folder_name] = counters.get(folder_name, 0) + 1
            ext      = Path(media_file).suffix
            img_name = f"img_{counters[folder_name]:03d}{ext}"
            out_path = out_dir / img_name

            zip_path = media_files[media_file]
            with z.open(zip_path) as src, open(out_path, "wb") as dst:
                shutil.copyfileobj(src, dst)

            records.append({
                "image_path": str(out_path.relative_to(img_dir.parent)),
                "label":      disease,
                "folder":     folder_name,
                "filename":   img_name,
                "source_rid": rid,
            })
            print(f"  OK {folder_name}/{img_name}  <- {disease}")

    return records


# ── Écriture du CSV ───────────────────────────────────────────────────────────

def write_csv(records, csv_path: Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "label", "folder", "filename", "source_rid"])
        writer.writeheader()
        writer.writerows(records)
    print(f"\nCSV écrit : {csv_path}  ({len(records)} entrées)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=== Analyse du document ===")
    paragraphs, rels, media_files = parse_document(DOCX_PATH)
    print(f"  Paragraphes : {len(paragraphs)}")
    print(f"  Médias dans le ZIP : {len(media_files)}")
    print(f"  Relations image : {len(rels)}\n")

    print("=== Association image -> maladie ===")
    labeled = build_label_map(paragraphs, rels, media_files)
    print(f"  Images labelisees : {len(labeled)}\n")

    # Nettoyage du dossier de sortie
    if IMG_DIR.exists():
        shutil.rmtree(IMG_DIR)

    print("=== Extraction des images ===")
    records = extract_images(DOCX_PATH, labeled, media_files, IMG_DIR)

    print("\n=== Bilan par maladie ===")
    from collections import Counter
    counts = Counter(r["label"] for r in records)
    for disease, count in sorted(counts.items()):
        print(f"  {disease:<40} {count} image(s)")

    unlabeled = len(media_files) - len(labeled)
    if unlabeled > 0:
        print(f"\n  ATTENTION: {unlabeled} image(s) dans le document sans label detecte")

    write_csv(records, CSV_OUT)
    print("\nTerminé.")


if __name__ == "__main__":
    main()
