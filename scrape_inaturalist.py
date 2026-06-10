"""
Scraping d'images via l'API iNaturalist (gratuit, sans clé API).
Cible : 27 maladies de plantes, 50-100 photos par classe.

iNaturalist = réseau mondial d'observations naturalistes avec photos géolocalisées.
API docs : https://api.inaturalist.org/v1/docs/
"""

import os
import time
import requests
from pathlib import Path

# ── Chemins ───────────────────────────────────────────────────────────────────
BASE_DIR  = Path(r"c:\Users\Larissa CONSEIGA\agrolaafi\plant-disease")
OUT_DIR   = BASE_DIR / "data" / "scraped" / "inaturalist"
LOG_FILE  = BASE_DIR / "data" / "scraped" / "inaturalist_log.csv"

# Limite d'images par classe (augmenter selon besoins)
MAX_PER_CLASS = 80

# Pause entre requêtes pour ne pas surcharger l'API (secondes)
REQUEST_DELAY = 0.5


# ── Mapping maladie → termes de recherche iNaturalist ────────────────────────
#
# Pour les maladies fongiques/bactériennes/virales : on cherche le pathogène
# Pour les ravageurs (insectes/acariens) : on cherche l'organisme directement
# Pour les désordres physiologiques : on cherche avec mots-clés généraux
#
DISEASE_QUERIES = {

    # ── Champignons ──────────────────────────────────────────────────────────
    "alternariose": {
        "taxon_name": "Alternaria solani",
        "label": "Alternariose",
    },
    "botrytis": {
        "taxon_name": "Botrytis cinerea",
        "label": "Botrytis",
    },
    "fusariose": {
        "taxon_name": "Fusarium oxysporum",
        "label": "Fusariose",
    },
    "mildiou": {
        "taxon_name": "Phytophthora infestans",
        "label": "Mildiou",
    },
    "oidium": {
        "taxon_name": "Leveillula taurica",
        "label": "Oidium",
    },
    "anthracnose": {
        "taxon_name": "Colletotrichum",
        "label": "Anthracnose",
    },
    "stemphyliose": {
        "taxon_name": "Stemphylium solani",
        "label": "Stemphyliose",
    },
    "verticilliose": {
        "taxon_name": "Verticillium dahliae",
        "label": "Verticilliose",
    },
    "fonte_de_semis": {
        "taxon_name": "Pythium",
        "label": "Fonte de semis",
    },

    # ── Bactéries ─────────────────────────────────────────────────────────────
    "fletrissement_bacterien": {
        "taxon_name": "Ralstonia solanacearum",
        "label": "Flétrissement bactérien",
    },
    "gale_bacterienne": {
        "taxon_name": "Clavibacter michiganensis",
        "label": "Gale bactérienne",
    },

    # ── Virus ─────────────────────────────────────────────────────────────────
    "tylcv": {
        "taxon_name": "Tomato yellow leaf curl virus",
        "label": "TYLCV",
    },
    "tmv": {
        "taxon_name": "Tobacco mosaic virus",
        "label": "TMV",
    },
    "cmv": {
        "taxon_name": "Cucumber mosaic virus",
        "label": "CMV",
    },

    # ── Insectes ──────────────────────────────────────────────────────────────
    "mineuse_de_la_tomate": {
        "taxon_name": "Tuta absoluta",
        "label": "Mineuse de la tomate",
    },
    "mouche_blanche": {
        "taxon_name": "Bemisia tabaci",
        "label": "Mouche blanche",
    },
    "noctuelle_de_la_tomate": {
        "taxon_name": "Helicoverpa armigera",
        "label": "Noctuelle de la tomate",
    },
    "puceron": {
        "taxon_name": "Myzus persicae",
        "label": "Puceron",
    },
    "thrips": {
        "taxon_name": "Frankliniella occidentalis",
        "label": "Thrips",
    },

    # ── Acariens ──────────────────────────────────────────────────────────────
    "araignee_rouge": {
        "taxon_name": "Tetranychus urticae",
        "label": "Araignée rouge / Tétranyque",
    },
    "acariose_bronzee": {
        "taxon_name": "Aculops lycopersici",
        "label": "Acariose bronzée",
    },
    "tarsoneme": {
        "taxon_name": "Polyphagotarsonemus latus",
        "label": "Tarsonème",
    },
    "nematodes": {
        "taxon_name": "Meloidogyne",
        "label": "Nématodes à galles",
    },

    # ── Désordres physiologiques / Carences ───────────────────────────────────
    # Note: ces catégories ne sont PAS dans iNaturalist (pas d'organisme vivant)
    # → seront gérées par Bing dans scrape_bing.py
}

# Classes à gérer séparément via Bing (pas de taxon sur iNaturalist)
BING_ONLY = ["carence_en_azote", "carence_en_potassium", "necrose_apicale", "coup_de_soleil"]


# ── Fonctions API ─────────────────────────────────────────────────────────────

def search_observations(taxon_name: str, page: int = 1, per_page: int = 100) -> dict:
    """Appelle l'API iNaturalist et retourne les observations avec photos."""
    url    = "https://api.inaturalist.org/v1/observations"
    params = {
        "taxon_name": taxon_name,
        "photos":     "true",
        "per_page":   per_page,
        "page":       page,
        "quality_grade": "research",   # observations validées par la communauté
        "order_by":   "votes",         # les meilleures photos en premier
    }
    headers = {"User-Agent": "AgroLaafi-DataCollection/1.0 (burkina-faso-plant-disease-research)"}
    resp = requests.get(url, params=params, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json()


def extract_photo_urls(observations: list) -> list[str]:
    """Extrait les URLs des photos en qualité 'large' (environ 1024px)."""
    urls = []
    for obs in observations:
        for photo in obs.get("photos", []):
            url = photo.get("url", "")
            if url:
                # iNaturalist renvoie des thumbnails — on prend la version large
                url = url.replace("square", "large").replace("thumb", "large")
                urls.append(url)
    return urls


def download_image(url: str, out_path: Path, session: requests.Session) -> bool:
    """Télécharge une image et la sauvegarde. Retourne True si succès."""
    try:
        resp = session.get(url, timeout=20, stream=True)
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "")
        if "image" not in content_type:
            return False
        with open(out_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        return out_path.stat().st_size > 5000   # ignore les fichiers trop petits
    except Exception:
        return False


# ── Pipeline principal ────────────────────────────────────────────────────────

def scrape_class(key: str, config: dict, session: requests.Session) -> int:
    """Scrape les images pour une classe de maladie. Retourne le nombre d'images."""
    label      = config["label"]
    taxon_name = config["taxon_name"]
    out_dir    = OUT_DIR / key
    out_dir.mkdir(parents=True, exist_ok=True)

    # Compte les images déjà téléchargées
    existing = len(list(out_dir.glob("*.jpg")))
    if existing >= MAX_PER_CLASS:
        print(f"  [SKIP] {label} — {existing} images déjà présentes")
        return existing

    needed = MAX_PER_CLASS - existing
    print(f"  {label} ({taxon_name}) — besoin de {needed} images ...")

    all_urls = []
    page = 1
    while len(all_urls) < needed + 10:
        try:
            data = search_observations(taxon_name, page=page, per_page=100)
        except requests.RequestException as e:
            print(f"    Erreur API page {page}: {e}")
            break

        obs      = data.get("results", [])
        if not obs:
            break
        urls = extract_photo_urls(obs)
        all_urls.extend(urls)
        total_available = data.get("total_results", 0)
        if len(all_urls) >= total_available or page >= 5:
            break
        page += 1
        time.sleep(REQUEST_DELAY)

    downloaded = existing
    counter    = existing + 1
    for url in all_urls:
        if downloaded >= MAX_PER_CLASS:
            break
        out_path = out_dir / f"inat_{counter:04d}.jpg"
        if download_image(url, out_path, session):
            downloaded += 1
            counter    += 1
            if downloaded % 10 == 0:
                print(f"    {downloaded}/{MAX_PER_CLASS} ...")
        time.sleep(REQUEST_DELAY * 0.5)

    gained = downloaded - existing
    print(f"    Resultat : {gained} nouvelles images (total {downloaded})")
    return downloaded


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    print("=== Scraping iNaturalist ===")
    print(f"Objectif : {MAX_PER_CLASS} images / classe")
    print(f"Classes  : {len(DISEASE_QUERIES)}")
    print(f"Sortie   : {OUT_DIR}\n")

    log_lines = ["classe,label,images_telechargees"]
    session   = requests.Session()
    session.headers["User-Agent"] = "AgroLaafi/1.0"

    total = 0
    for key, config in DISEASE_QUERIES.items():
        count = scrape_class(key, config, session)
        log_lines.append(f"{key},{config['label']},{count}")
        total += count
        time.sleep(1.0)

    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))

    print(f"\nTotal images scrappees : {total}")
    print(f"Log                   : {LOG_FILE}")
    print(f"\nNote : {len(BING_ONLY)} classes a completer via scrape_bing.py :")
    for c in BING_ONLY:
        print(f"  - {c}")


if __name__ == "__main__":
    main()
