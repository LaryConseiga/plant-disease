"""
Scraping d'images via l'API Wikimedia Commons (gratuit, sans clé, stable).
Utilisé pour les 4 désordres physiologiques non couverts par iNaturalist :
  - Carence en azote
  - Carence en potassium
  - Nécrose apicale
  - Coup de soleil

API Wikimedia : https://commons.wikimedia.org/w/api.php
Licence : images sous Creative Commons (libre d'utilisation pour la recherche)
"""

import time
import requests
from pathlib import Path

# ── Chemins ───────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
OUT_DIR  = BASE_DIR / "data" / "scraped" / "wikimedia"

MAX_PER_CLASS = 60
DELAY         = 2.0   # secondes entre requêtes (Wikimedia rate limit)
MAX_RETRIES   = 3

API_URL = "https://commons.wikimedia.org/w/api.php"
# User-Agent conforme à la politique Wikimedia (contact requis)
HEADERS = {
    "User-Agent": "AgroLaafi-DataCollection/1.0 (https://github.com/agrolaafi; larissacherifa7@gmail.com) Python/3.13"
}


# ── Requêtes par classe ───────────────────────────────────────────────────────
DISEASE_QUERIES = {
    "carence_en_azote": {
        "label":   "Carence en azote",
        "queries": [
            "nitrogen deficiency plant",
            "nitrogen deficiency tomato",
            "carence azote plante",
        ],
    },
    "carence_en_potassium": {
        "label":   "Carence en potassium",
        "queries": [
            "potassium deficiency plant",
            "potassium deficiency tomato",
            "carence potassium",
        ],
    },
    "necrose_apicale": {
        "label":   "Nécrose apicale",
        "queries": [
            "blossom end rot tomato",
            "blossom end rot",
            "necrose apicale tomate",
        ],
    },
    "coup_de_soleil": {
        "label":   "Coup de soleil",
        "queries": [
            "sunscald tomato",
            "sunburn plant fruit",
            "coup soleil legume",
        ],
    },
}


# ── API Wikimedia ─────────────────────────────────────────────────────────────

def search_images(query: str, session: requests.Session, limit: int = 50) -> list[str]:
    """
    Cherche des images sur Wikimedia Commons.
    Retourne une liste de noms de fichiers (ex: 'File:Blossom_end_rot.jpg').
    """
    params = {
        "action":      "query",
        "list":        "search",
        "srsearch":    query,
        "srnamespace": 6,
        "srlimit":     limit,
        "format":      "json",
    }
    for attempt in range(MAX_RETRIES):
        try:
            resp = session.get(API_URL, params=params, timeout=30)
            if resp.status_code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            resp.raise_for_status()
            results = resp.json().get("query", {}).get("search", [])
            return [r["title"] for r in results]
        except (requests.exceptions.Timeout,
                requests.exceptions.ConnectionError,
                requests.exceptions.RequestException):
            time.sleep(3 * (attempt + 1))
    return []


def get_image_url(file_title: str, session: requests.Session) -> str | None:
    """
    Récupère l'URL directe d'un fichier Wikimedia.
    Ex: 'File:Blossom_end_rot.jpg' → 'https://upload.wikimedia.org/...'
    Retourne None en cas d'erreur réseau (timeout, connexion refusée).
    """
    params = {
        "action":     "query",
        "titles":     file_title,
        "prop":       "imageinfo",
        "iiprop":     "url|mime|size",
        "iiurlwidth": 800,
        "format":     "json",
    }
    for attempt in range(MAX_RETRIES):
        try:
            resp = session.get(API_URL, params=params, timeout=30)
            if resp.status_code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            resp.raise_for_status()
            pages = resp.json().get("query", {}).get("pages", {})
            for page in pages.values():
                info = page.get("imageinfo", [{}])[0]
                mime = info.get("mime", "")
                if "image" in mime and "svg" not in mime:
                    return info.get("thumburl") or info.get("url")
            return None
        except (requests.exceptions.Timeout,
                requests.exceptions.ConnectionError,
                requests.exceptions.RequestException):
            time.sleep(3 * (attempt + 1))
    return None


def download_image(url: str, out_path: Path, session: requests.Session) -> bool:
    """Télécharge une image avec retry sur 429. Retourne True si succès."""
    for attempt in range(MAX_RETRIES):
        try:
            resp = session.get(url, timeout=30, stream=True)
            if resp.status_code == 429:
                wait = 5 * (attempt + 1)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            if "image" not in resp.headers.get("content-type", ""):
                return False
            with open(out_path, "wb") as f:
                for chunk in resp.iter_content(8192):
                    f.write(chunk)
            return out_path.stat().st_size > 5000
        except requests.RequestException:
            time.sleep(3)
    return False


# ── Pipeline ──────────────────────────────────────────────────────────────────

def scrape_class(key: str, config: dict, session: requests.Session) -> int:
    label   = config["label"]
    queries = config["queries"]
    out_dir = OUT_DIR / key
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = len(list(out_dir.glob("*.jpg")) + list(out_dir.glob("*.png")))
    if existing >= MAX_PER_CLASS:
        print(f"  [SKIP] {label} — {existing} images deja presentes")
        return existing

    print(f"  {label} ...")

    all_files = []
    for query in queries:
        files = search_images(query, session, limit=50)
        all_files.extend(files)
        time.sleep(DELAY)

    # Dédoublonnage
    seen      = set()
    all_files = [f for f in all_files if not (f in seen or seen.add(f))]

    downloaded = existing
    counter    = existing + 1

    for file_title in all_files:
        if downloaded >= MAX_PER_CLASS:
            break
        url = get_image_url(file_title, session)
        time.sleep(DELAY * 0.5)
        if not url:
            continue
        ext      = ".jpg" if "jpg" in url.lower() else ".png"
        out_path = out_dir / f"wiki_{counter:04d}{ext}"
        if download_image(url, out_path, session):
            downloaded += 1
            counter    += 1
        time.sleep(DELAY)

    gained = downloaded - existing
    print(f"    -> {gained} nouvelles images (total {downloaded})")
    return downloaded


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=== Scraping Wikimedia Commons ===")
    print(f"Classes : {list(DISEASE_QUERIES.keys())}\n")

    session = requests.Session()
    session.headers.update(HEADERS)

    total = 0
    for key, config in DISEASE_QUERIES.items():
        count = scrape_class(key, config, session)
        total += count

    print(f"\nTotal : {total} images")
    print(f"Sortie : {OUT_DIR}")


if __name__ == "__main__":
    main()
