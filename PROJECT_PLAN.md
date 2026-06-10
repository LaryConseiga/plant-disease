# AgroLaafi Project Plan

## 1. Project Overview

AgroLaafi est une solution agricole burkinabè basée sur l'IA qui détecte les maladies des plantes à partir d'une photo. Le projet inclut un produit matériel (`Laafi Box`) et une application mobile, avec une attention particulière sur l'usage hors-ligne, la géolocalisation, et l'assistance vocale en langues locales (mooré, dioula, fulfuldé, français).

Objectifs principaux:

- Diagnostiquer les maladies des cultures par photo
- Fournir des conseils et traitements locaux
- Offrir une interface vocale adaptée aux agriculteurs sans formation technique
- Construire une base de données géolocalisée pour suivi épidémiologique

## 2. Documents sources

- `AGROLAAFI.pptx`: Concept, fonctionnalités, innovation, faisabilité technique, opportunité marché, équipe.
- `AgroLaafi_Business_Plan.pptx`: Business plan, produit, marché, revenus, feuille de route, impact.
- `Maladie word.docx`: Fiches maladies / symptômes / lutte pour de nombreuses maladies des tomates et autres cultures.

## 3. Travail à organiser

### Phase 1: Clarification du périmètre

- Définir le MVP clair pour commencer:
  - Détection photo d'une ou plusieurs cultures prioritaires (ex: tomate)
  - Identification des maladies les plus communes listées dans le docx
  - Interface simple mobile ou prototype web
- Définir les classes de maladies à inclure initialement en se basant sur le docx.
- Choisir la plateforme de déploiement initiale: mobile Android, web, ou prototype local.

### Phase 2: Données et annotation

- Extraire la taxonomie des maladies du docx:
  - Acariose bronzée
  - Alternariose
  - Anthracnose
  - Araignée rouge / Tétranyque
  - Botrytis
  - Carences (azote, potassium)
  - CMV, TMV, TYLCV
  - Verticilliose, Fusariose, Mildiou, Oidium
  - Mineuse de la tomate, Mouche blanche, Thrips, etc.
- Catégoriser les types:
  - Champignons
  - Virus
  - Bactéries
  - Acariens / insectes
  - Carences nutritionnelles
  - Physiologiques (coup de soleil)
- Créer un fichier `data/labels.csv` ou `data/disease_taxonomy.md` avec:
  - nom de la maladie
  - culture cible
  - symptômes principaux
  - type de pathogène
  - traitement recommandé
- Collecter ou acheter des images labellisées pour chaque maladie ciblée.
- Si possible, collecter des images de référence pour les cultures locales au Burkina Faso.

### Phase 3: Pipeline IA

- Choisir une architecture de modèle pour classification d'images:
  - transfer learning CNN (MobileNet, EfficientNet, ResNet, etc.)
  - TensorFlow / PyTorch
- Construire le pipeline:
  - `data/raw/` pour images brutes
  - `data/processed/` pour images redimensionnées et augmentées
  - `src/model/train.py` pour entraînement
  - `src/model/evaluate.py` pour métriques
- Prévoir l'évaluation avec des métriques claires:
  - précision, rappel, matrice de confusion
  - performance par classe de maladie
- Tester la robustesse en conditions réelles (lumière, feuilles sales, angles variés).

### Phase 4: Produit et intégration

- Définir le MVP produit:
  - prise de photo
  - prédiction maladie
  - affichage de symptômes et solution locale
- Si la cible est mobile:
  - utiliser TensorFlow Lite ou un modèle Edge
  - interface simple avec voix si possible
- Si la cible est prototype web:
  - API backend pour l’inférence
  - frontend simple avec upload photo
- Documenter l'architecture du produit:
  - `Laafi Box` = modèle embarqué + caméra + énergie solaire
  - app mobile = diagnostic, géodatabase, communauté
  - backend = synchronisation, base de données locale

### Phase 5: Business / roadmap

- Reprendre la feuille de route du business plan:
  - Phase 1: Prototype & 150 unités test
  - Phase 2: Application mobile + géodatabase
  - Phase 3: Extension régionale
- Prioriser les livrables techniques sur 12–24 mois:
  - prototype IA fonctionnel
  - dataset maladies locales
  - démonstrateur mobile/web
  - preuves terrain avec coopératives

## 4. Recommandation de structure de projet

- `data/`
  - `raw/`
  - `processed/`
  - `labels/`
- `docs/`
  - `business_plan/`
  - `disease_reference.md`
- `notebooks/`
  - `exploration.ipynb`
  - `training.ipynb`
- `src/`
  - `model/`
  - `app/`
  - `data/`
- `PROJECT_PLAN.md`
- `README.md`

## 5. Prochaines actions immédiates

1. Extraire une liste de maladies et labels précis depuis `Maladie word.docx`.
2. Choisir un sous-ensemble de cultures/maladies pour le MVP.
3. Collecter ou préparer un dataset d’images pour ces classes.
4. Prototyper un modèle de classification image simple et tester avec des images locales.
5. Préparer un document de spécification technique minimal pour le MVP.

## 6. Conseils pratiques

- Commencez petit: un modèle fiable sur un petit nombre de maladies vaut mieux qu’un modèle fragile sur trop de classes.
- Concentrez-vous sur les cultures du Burkina Faso les plus importantes.
- Utilisez les descriptions du docx pour construire des labels utiles et des explications métier.
- Documentez clairement chaque dataset, chaque expérience de modèle, et les sources de données.

---

_Ce plan est basé sur le contenu de vos PPTX et DOCX. Je peux aussi vous aider à transformer le docx en un fichier `data/disease_taxonomy.md` ou à définir un pipeline de données plus détaillé._
