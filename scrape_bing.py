"""
Scraping d'images via DuckDuckGo Images (sans clé API, sans compte).
Utilisé pour :
  1. Compléter les classes où iNaturalist n'a pas assez de photos
  2. Les désordres physiologiques/carences (pas dans iNaturalist)
  3. Les 5 classes disponibles dans PlantVillage (alternative directe)

Nécessite : pip install icrawler
"""

import os
import time
import shutil
from pathlib import Path

# ── Vérification de la dépendance ─────────────────────────────────────────────
try:
    from icrawler.builtin import BingImageCrawler, GoogleImageCrawler
    ICRAWLER_OK = True
except ImportError:
    ICRAWLER_OK = False
    print("icrawler non installe. Lancez : py -m pip install icrawler")
    exit(1)

# ── Chemins ───────────────────────────────────────────────────────────────────
BASE_DIR = Path(r"c:\Users\Larissa CONSEIGA\agrolaafi\plant-disease")
OUT_DIR  = BASE_DIR / "data" / "scraped" / "bing"

# Images à télécharger par classe
MAX_PER_CLASS = 80


# ── Requêtes de recherche ─────────────────────────────────────────────────────
#
# Stratégie : combiner nom français + anglais + nom scientifique
# pour maximiser la pertinence des résultats
#
SEARCH_QUERIES = {

    # ── Désordres physiologiques / Carences (absents d'iNaturalist) ───────────
    "carence_en_azote": {
        "label":   "Carence en azote",
        "queries": [
            "nitrogen deficiency tomato leaves symptoms",
            "carence azote tomate symptomes feuilles jaunes",
        ],
    },
    "carence_en_potassium": {
        "label":   "Carence en potassium",
        "queries": [
            "potassium deficiency tomato symptoms",
            "carence potassium tomate necrose marginelle",
        ],
    },
    "necrose_apicale": {
        "label":   "Nécrose apicale",
        "queries": [
            "blossom end rot tomato symptoms",
            "necrose apicale tomate calcium deficiency",
        ],
    },
    "coup_de_soleil": {
        "label":   "Coup de soleil",
        "queries": [
            "sunscald tomato fruit symptoms",
            "coup de soleil tomate fruit blanc",
        ],
    },

    # ── Classes PlantVillage (complément à ce que iNaturalist trouve) ─────────
    "alternariose": {
        "label":   "Alternariose",
        "queries": [
            "Alternaria solani tomato early blight symptoms leaves",
            "alternariose tomate taches brunes halo jaune",
        ],
    },
    "mildiou": {
        "label":   "Mildiou",
        "queries": [
            "Phytophthora infestans tomato late blight symptoms",
            "mildiou tomate taches aqueuses brun",
        ],
    },
    "araignee_rouge": {
        "label":   "Araignée rouge / Tétranyque",
        "queries": [
            "Tetranychus urticae tomato spider mites symptoms",
            "araignee rouge tomate tetranyque toile",
        ],
    },
    "tmv": {
        "label":   "TMV",
        "queries": [
            "tobacco mosaic virus tomato symptoms mosaic",
            "virus mosaique tabac tomate TMV symptoms",
        ],
    },
    "tylcv": {
        "label":   "TYLCV",
        "queries": [
            "tomato yellow leaf curl virus symptoms",
            "TYLCV tomate enroulement foliaire jaunissement",
        ],
    },

    # ── Maladies peu représentées sur iNaturalist ─────────────────────────────
    "acariose_bronzee": {
        "label":   "Acariose bronzée",
        "queries": [
            "Aculops lycopersici tomato russet mite bronze symptoms",
            "acariose bronze tomate tige brun rouille",
        ],
    },
    "fonte_de_semis": {
        "label":   "Fonte de semis",
        "queries": [
            "damping off seedlings tomato Pythium symptoms",
            "fonte semis tomate plantules flétrissement",
        ],
    },
    "botrytis": {
        "label":   "Botrytis",
        "queries": [
            "Botrytis cinerea tomato grey mold symptoms",
            "botrytis tomate duvet gris pourriture",
        ],
    },
    "anthracnose": {
        "label":   "Anthracnose",
        "queries": [
            "Colletotrichum tomato anthracnose fruit rot symptoms",
            "anthracnose tomate taches sombres fruit",
        ],
    },
    "fusariose": {
        "label":   "Fusariose",
        "queries": [
            "Fusarium oxysporum tomato fusarium wilt symptoms",
            "fusariose tomate fletrissement xyleme brun",
        ],
    },
    "fletrissement_bacterien": {
        "label":   "Flétrissement bactérien",
        "queries": [
            "Ralstonia solanacearum tomato bacterial wilt symptoms",
            "fletrissement bacterien tomate Ralstonia exsudat",
        ],
    },
    "gale_bacterienne": {
        "label":   "Gale bactérienne",
        "queries": [
            "Clavibacter michiganensis tomato bacterial canker symptoms",
            "gale bacterienne tomate chancre clavibacter",
        ],
    },
    "mineuse_de_la_tomate": {
        "label":   "Mineuse de la tomate",
        "queries": [
            "Tuta absoluta tomato leafminer damage symptoms",
            "mineuse tomate Tuta absoluta galeries feuilles",
        ],
    },
    "mouche_blanche": {
        "label":   "Mouche blanche",
        "queries": [
            "Bemisia tabaci whitefly tomato damage symptoms",
            "aleurode mouche blanche tomate fumagine miellat",
        ],
    },
    "noctuelle": {
        "label":   "Noctuelle de la tomate",
        "queries": [
            "Helicoverpa armigera tomato fruitworm damage",
            "noctuelle tomate helicoverpa larve fruit perfore",
        ],
    },
    "puceron": {
        "label":   "Puceron",
        "queries": [
            "Myzus persicae aphid tomato damage symptoms",
            "puceron tomate colonies feuilles enroulees",
        ],
    },
    "thrips": {
        "label":   "Thrips",
        "queries": [
            "Frankliniella occidentalis tomato thrips damage silver leaves",
            "thrips tomate feuilles argentees deformation",
        ],
    },
    "oidium": {
        "label":   "Oïdium",
        "queries": [
            "Leveillula taurica tomato powdery mildew symptoms",
            "oidium tomate poudre blanche champignon",
        ],
    },
    "nematodes": {
        "label":   "Nématodes à galles",
        "queries": [
            "Meloidogyne root knot nematode tomato symptoms",
            "nematodes galles tomate racines nodosites",
        ],
    },
    "cmv": {
        "label":   "CMV",
        "queries": [
            "cucumber mosaic virus tomato symptoms mosaic deformation",
            "CMV tomate mosaique deformation feuilles",
        ],
    },
    "stemphyliose": {
        "label":   "Stemphyliose",
        "queries": [
            "Stemphylium solani tomato leaf blight symptoms gray brown",
            "stemphyliose tomate taches grises brunes feuilles",
        ],
    },
    "tarsoneme": {
        "label":   "Tarsonème",
        "queries": [
            "Polyphagotarsonemus latus broad mite tomato symptoms",
            "tarsoneme tomate feuilles deformees cafe",
        ],
    },
    "verticilliose": {
        "label":   "Verticilliose",
        "queries": [
            "Verticillium dahliae tomato wilt symptoms yellowing",
            "verticilliose tomate flétrissement nervures jaunes",
        ],
    },
}


# ── Téléchargement ────────────────────────────────────────────────────────────

def crawl_class(key: str, config: dict) -> int:
    """Télécharge les images pour une classe via Bing."""
    label   = config["label"]
    queries = config["queries"]
    out_dir = OUT_DIR / key
    out_dir.mkdir(parents=True, exist_ok=True)

    # Compte les images déjà là
    existing = len(list(out_dir.glob("*.jpg")) + list(out_dir.glob("*.png")))
    if existing >= MAX_PER_CLASS:
        print(f"  [SKIP] {label} — {existing} images déjà présentes")
        return existing

    per_query = max(1, (MAX_PER_CLASS - existing) // len(queries) + 5)
    counter   = existing

    for query in queries:
        if counter >= MAX_PER_CLASS:
            break
        tmp_dir = out_dir / "_tmp"
        tmp_dir.mkdir(exist_ok=True)

        crawler = BingImageCrawler(
            downloader_threads=4,
            storage={"root_dir": str(tmp_dir)},
        )
        # Désactive les logs internes du crawler
        import logging
        logging.getLogger("icrawler").setLevel(logging.ERROR)

        try:
            crawler.crawl(keyword=query, max_num=per_query, min_size=(100, 100))
        except Exception as e:
            print(f"    Erreur crawl '{query}': {e}")

        # Renomme et déplace les fichiers téléchargés
        for f in sorted(tmp_dir.glob("*")):
            if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
                counter += 1
                dest = out_dir / f"bing_{counter:04d}{f.suffix.lower()}"
                shutil.move(str(f), str(dest))
                if counter >= MAX_PER_CLASS:
                    break

        # Nettoyage tmp
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)

        time.sleep(1.0)

    gained = counter - existing
    print(f"  {label:<40} {gained:>3} nouvelles images (total {counter})")
    return counter


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Scraping Bing Images ===")
    print(f"Objectif : {MAX_PER_CLASS} images / classe")
    print(f"Classes  : {len(SEARCH_QUERIES)}")
    print(f"Sortie   : {OUT_DIR}\n")

    total = 0
    for key, config in SEARCH_QUERIES.items():
        count = crawl_class(key, config)
        total += count

    print(f"\nTotal images telechargees : {total}")
    print(f"Dossier : {OUT_DIR}")


if __name__ == "__main__":
    main()
