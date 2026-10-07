# Dossier RH — Knowledge Graph & GraphRAG CV Ranking Engine

Système d'évaluation et de classement de candidats fondé sur un **graphe de connaissances ontologique (RDFLib & SKOS)**, une **analyse topologique avancée (NetworkX, Personalized PageRank, distances taxopathiques)**, une **détection automatisée des contradictions et incohérences temporelles**, et une **synthèse qualitative GraphRAG par communautés de Louvain**.

---

## 1. Vue d'ensemble de l'architecture

Le moteur dépasse les limites des approches purement vectorielles ou par mots-clés en intégrant une modélisation sémantique stricte et une validation d'intégrité avant calcul du score.

```
src/
├── core/
│   ├── ontology.py          # Ontologie RDFLib (namespace CV), taxonomies SKOS et distances taxopathiques
│   ├── models.py            # Schémas Pydantic v2 stricts : Entity, Relation, Payloads, Conflict, ValidationReport
│   └── validator.py         # Moteur de validation : inversions temporelles, anachronismes, sur-déclarations
├── graph/
│   ├── builder.py           # Ingestion triplets RDFLib & NetworkX (MultiDiGraph) bidirectionnelle
│   ├── matching.py          # Scoring topologique : Personalized PageRank (PPR), similarité SKOS, Jaccard
│   └── graph_rag.py         # Clustering par communautés Louvain & génération de critiques explicables
├── pipeline/
│   ├── extractor.py         # Extraction de relations & qualificatifs temporels (LLM + parser déterministe)
│   └── ranker.py            # Scoring hybride multi-facteurs (Vecteurs + Graphe - Pénalités)
├── extraction/              # Extracteur historique regex (rétro-compatibilité)
├── parsing/                 # Parsing de documents PDF (pdfplumber), DOCX (python-docx), TXT
└── scoring/                 # Moteur de scoring de base
tests/
├── test_contradictions.py   # Tests d'inversions, anachronismes et sur-déclarations
├── test_graph_builder.py    # Tests d'ingestion RDFLib, conversion NetworkX et réification d'arêtes
└── test_ranking.py          # Tests de PPR, crédit partiel taxonomique, ranker hybride et GraphRAG
```

---

## 2. Pipeline de traitement & Fonctionnalités clés

```
CV & Offre d'emploi (PDF, DOCX, TXT)
            │
            ▼
┌────────────────────────────────────────────────────────┐
│ 1. Extraction Structurée des Triplets (S-P-O)          │
│    - Entités : CANDIDATE, ROLE, SKILL, DEGREE, COMPANY │
│    - Relations qualifiées : dates, durées, dépendances │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ 2. Validation Pré-Scoring & Détection de Conflits      │
│    - Inversions temporelles (start_date > end_date)     │
│    - Anachronismes (usage techno < date de sortie)      │
│    - Sur-déclaration cumulative (années > contrats)    │
│    ➔ ValidationReport(conflicts, penalty_factor)       │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ 3. Représentation Ontologique (RDFLib & NetworkX)      │
│    - Schéma RDF/SKOS (CV.Skill, skos:broader, etc.)     │
│    - Conversion bidirectionnelle vers nx.MultiDiGraph  │
│    - Réification des arêtes (dates, confiance)         │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ 4. Analyse Topologique & Scoring GraphRAG              │
│    - Personalized PageRank (PPR) diffusé depuis l'offre│
│    - Crédit partiel pour compétences proches (SKOS)    │
│    - Détection de communautés (Louvain) & Synthèse     │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ 5. Synthèse & Classement Multi-Facteurs                │
│    FinalScore = α·Vector + β·GraphMatch - γ·Penalty    │
│    Tableau de classement + Audit d'explicabilité       │
└────────────────────────────────────────────────────────┘
```

---

## 3. Formule de Scoring Hybride

Le classement des candidats repose sur une formulation objective et configurable :

$$\text{FinalScore} = \alpha \cdot \text{VectorSimilarity} + \beta \cdot \text{GraphMatchScore} - \gamma \cdot \text{ConflictPenalty}$$

| Composante | Rôle & Méthode de Calcul |
|---|---|
| **$\text{VectorSimilarity}$** | Similarité textuelle TF-IDF et recouvrement lexical entre le profil et l'offre d'emploi. |
| **$\text{GraphMatchScore}$** | Synthèse topologique combinant :<br>• **Personalized PageRank (PPR)** : diffusion d'importance depuis les compétences requises par l'offre vers le nœud candidat.<br>• **Distance Taxopathique SKOS** : attribution d'un crédit partiel pour les technologies parentes ou connexes (ex. *PyTorch* ➔ *Deep Learning* = 1 saut, *TensorFlow* = 2 sauts).<br>• **Similarité de Jaccard** : taux de recouvrement strict des compétences. |
| **$\text{ConflictPenalty}$** | Pénalité déductive issue du `ValidationReport` proportionnelle à la sévérité des anomalies détectées. |

---

## 4. Détection Automatisée des Incohérences

Avant toute étape de scoring, le module `ConsistencyValidator` applique des règles de cohérence :
1. **Inversions temporelles** : détection des relations où la date de début est postérieure à la date de fin (`start_date > end_date`).
2. **Anachronismes technologiques** : vérification de la date de première utilisation déclarée par rapport au registre officiel des versions (ex. déclarer l'utilisation de *FastAPI* en 2014 alors qu'il est sorti en 2018).
3. **Sur-déclaration cumulative** : union des intervalles d'emplois documentés pour vérifier si le volume d'années déclaré sur une compétence est effectivement étayé par les postes occupés.
4. **Vérification d'hallucination** : contrôle des entités extraites par rapport aux limites textuelles du document source.

---

## 5. Synthèse GraphRAG & Détection de Communautés

Le module `GraphRAGSummaryEngine` :
- Segmente le sous-graphe du candidat en modules cohérents grâce à l'algorithme de **modularité de Louvain** (`networkx.algorithms.community.louvain_communities`).
- Identifie les domaines d'expertise dominants (ex. *AI & Machine Learning*, *Architecture Backend & APIs*, *Cloud & Infrastructure*).
- Génère un audit qualitatif explicable (`generate_candidate_critique`) détaillant :
  - Les compétences validées directement vs déduites par ontologie.
  - La continuité de la trajectoire professionnelle.
  - Le résumé exhaustif des alertes temporelles et de cohérence.

---

## 6. Installation & Prérequis

Prérequis : **Python ≥ 3.10**.

```bash
# 1. Cloner ou ouvrir le projet
cd cv-ranker

# 2. Créer et activer l'environnement virtuel
python -m venv .venv
# Sur Windows :
.venv\Scripts\activate
# Sur Linux/macOS :
source .venv/bin/activate

# 3. Installer les dépendances
pip install -r requirements.txt
```

### Configuration des variables d'environnement (optionnel)
Créez un fichier `.env` si vous souhaitez utiliser l'extraction LLM via Groq :
```env
GROQ_API_KEY=votre_cle_api_groq
```
*Note : Si aucune clé n'est fournie, le système bascule automatiquement et de manière transparente sur le parseur déterministe hors ligne.*

---

## 7. Exécution des Tests

La suite de tests unitaires valide l'ensemble du pipeline (détection d'incohérences, graphe RDFLib/NetworkX, scoring topologique et GraphRAG) :

```bash
python -m pytest tests/test_contradictions.py tests/test_graph_builder.py tests/test_ranking.py -v
```

Exemple de sortie :
```
tests/test_contradictions.py::test_clean_candidate_passes_validation PASSED
tests/test_contradictions.py::test_temporal_inversion_detected PASSED
tests/test_contradictions.py::test_anachronism_detected PASSED
tests/test_contradictions.py::test_cumulative_overclaiming_detected PASSED
tests/test_contradictions.py::test_multiple_contradictions_accumulate_penalty PASSED
tests/test_graph_builder.py::test_payload_to_rdflib_ingestion PASSED
tests/test_graph_builder.py::test_rdflib_to_networkx_preserves_attributes PASSED
tests/test_graph_builder.py::test_bidirectional_conversion_networkx_to_rdflib PASSED
tests/test_ranking.py::test_ontological_distance_and_partial_credit PASSED
tests/test_ranking.py::test_clean_candidate_ranking_flow PASSED
tests/test_ranking.py::test_contradictory_candidate_receives_penalty PASSED
tests/test_ranking.py::test_ppr_diffusion_and_community_critique PASSED
tests/test_ranking.py::test_extractor_end_to_end_parsing PASSED
tests/test_ranking.py::test_multiple_candidate_ranking_order PASSED

============================= 14 passed in 3.54s ==============================
```

---

## 8. Lancer l'Interface Streamlit

```bash
streamlit run app.py
```
Accédez ensuite à `http://localhost:8501` pour tester l'application interactive.

---

## 9. Exemple d'Utilisation en Python

```python
from src.pipeline.extractor import KnowledgeGraphExtractor
from src.pipeline.ranker import HybridRanker
from src.graph.graph_rag import GraphRAGSummaryEngine

# 1. Extraction des graphes candidat et offre
extractor = KnowledgeGraphExtractor(use_llm=False)
cv_graph = extractor.extract_candidate(cv_text, candidate_id="cand_1", candidate_name="Dr. Alan Turing")
jd_graph = extractor.extract_job(job_text, job_id="job_1", title="Lead AI Engineer")

# 2. Classement et calcul du score hybride
ranker = HybridRanker()
result = ranker.score_candidate(cv_graph, jd_graph)

print(f"Score final : {result.final_score}/100")
print(f"PPR Score : {result.graph_breakdown.ppr_score}")
print(f"Score taxonomique SKOS : {result.graph_breakdown.ontological_distance_score}")
print(f"Pénalité conflits : {result.conflict_penalty}")

# 3. Synthèse GraphRAG et audit explicable
engine = GraphRAGSummaryEngine(use_llm=False)
critique = engine.generate_candidate_critique(cv_graph, jd_graph)
print(critique)
```
