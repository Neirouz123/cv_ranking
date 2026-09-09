# Dossier RH — Classement de candidats (CV vs Offre)

Application Streamlit qui compare une offre d'emploi à un ou plusieurs CV
(PDF, DOCX ou texte) et calcule un **score de correspondance objectif et
explicable** pour chaque candidat, via un algorithme de pondération — sans
appel à un modèle de langage : chaque score est traçable jusqu'à une règle
précise (compétence détectée, seuil d'expérience, niveau de diplôme...).

## Architecture

```
cv-ranker/
├── app.py                          # Interface Streamlit (point d'entrée)
├── requirements.txt
├── .streamlit/config.toml          # Thème
├── src/
│   ├── parsing/
│   │   └── document_parser.py      # Extraction de texte : PDF (pdfplumber),
│   │                                #   DOCX (python-docx), TXT
│   ├── extraction/
│   │   ├── skills_data.py          # Taxonomie de compétences FR/EN (éditable)
│   │   └── extractor.py            # Détection de compétences, années
│   │                                #   d'expérience, niveau de formation
│   └── scoring/
│       └── scorer.py               # Algorithme de scoring pondéré
└── tests/                          # (voir section Tests)
```

### Flux de traitement

```
Offre d'emploi (texte / PDF / DOCX)          CV #1, CV #2, … (PDF / DOCX / TXT)
        │                                              │
        ▼                                              ▼
  document_parser.extract_text()             document_parser.extract_text()
        │                                              │
        ▼                                              ▼
  extractor.ExtractedProfile.from_text()  ──►  extractor.ExtractedProfile.from_text()
        │  (compétences, expérience,                    │
        │   niveau de formation)                        │
        └──────────────────┬─────────────────────────────┘
                            ▼
                 scorer.score_candidate(cv, offre, poids)
                            │
                            ▼
        ScoreBreakdown : score global + 4 sous-scores
        + compétences correspondantes / manquantes
        + points forts / points faibles générés par règles
                            │
                            ▼
              app.py : tableau de classement + détail par candidat
```

## Algorithme de scoring

Le score global (0–100) combine quatre composantes indépendantes,
pondérables depuis la barre latérale de l'application :

| Composante              | Poids par défaut | Calcul                                                              |
|--------------------------|:---:|-----------------------------------------------------------------------------|
| **Compétences**          | 50 % | % des compétences requises par l'offre retrouvées dans le CV (taxonomie de ~120 compétences FR/EN dans `skills_data.py`) |
| **Pertinence textuelle** | 20 % | Similarité cosinus TF-IDF entre le texte complet du CV et de l'offre (capte le vocabulaire métier hors taxonomie) |
| **Expérience**           | 15 % | Années d'expérience extraites du CV vs. seuil requis dans l'offre (règles regex FR/EN) |
| **Formation**            | 15 % | Niveau de diplôme détecté (Bac → Doctorat) vs. niveau requis          |

Chaque sous-score est calculé de façon déterministe et documentée dans
`scorer.py` — aucune "boîte noire" : un score peut toujours être justifié
auprès d'un candidat ou d'un manager.

**Limites connues (documentées dans le code) :**
- L'extraction d'expérience repose sur des formulations explicites
  ("5 ans d'expérience") plutôt que sur le calcul des dates d'emploi.
- La taxonomie de compétences est curatée manuellement ; elle est conçue
  pour être étendue facilement (ajout d'une clé dans `skills_data.py`).
- Les PDF scannés sans couche de texte (images) ne peuvent pas être lus
  sans étape d'OCR supplémentaire (non incluse).

## Installation

Prérequis : Python ≥ 3.10.

```bash
cd cv-ranker
python3 -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

## Lancer l'application

```bash
streamlit run app.py
```

Puis ouvrez l'URL affichée (par défaut http://localhost:8501).

## Utilisation

1. Collez le texte de l'offre d'emploi, ou importez-la en PDF/DOCX/TXT.
2. Importez un ou plusieurs CV (PDF, DOCX ou TXT).
3. Ajustez si besoin la pondération des critères dans la barre latérale.
4. Cliquez sur **Lancer le classement**.
5. Consultez le tableau de classement, exportez-le en CSV, et dépliez chaque
   candidat pour voir le détail des points forts / points faibles.

## Étendre la taxonomie de compétences

Ouvrez `src/extraction/skills_data.py` et ajoutez une entrée dans la
catégorie appropriée :

```python
"Nom affiché de la compétence": ["synonyme1", "synonyme 2", "abréviation"],
```

Les motifs sont des expressions régulières (insensibles à la casse et aux
accents) : les caractères spéciaux regex (`.`, `+`, `#`...) doivent être
échappés avec `\\`.

## Tests

Un test de bout en bout rapide (extraction + scoring, sans dépendance à
Streamlit) peut être exécuté directement :

```bash
python3 -c "
from src.extraction.extractor import ExtractedProfile
from src.scoring.scorer import score_candidate
job = ExtractedProfile.from_text(open('exemple_offre.txt').read())
cv = ExtractedProfile.from_text(open('exemple_cv.txt').read())
print(score_candidate(cv, job))
"
```

## Évolutions possibles

- Extraction de dates d'emploi (au lieu de la seule mention explicite
  des années d'expérience) pour un calcul plus robuste.
- OCR pour les CV scannés (ex. `pytesseract`).
- Détection de la localisation du candidat / contrainte de mobilité.
- Export PDF du dossier de synthèse par candidat.
