# Dossier RH — Knowledge Graph, SPARQL Audit & GraphRAG CV Ranking Engine

Système d'évaluation et de classement de candidats fondé sur un **graphe de connaissances ontologique (RDFLib, SKOS & SPARQL)**, une **analyse topologique avancée (NetworkX, Personalized PageRank, distances taxopathiques)**, une **détection automatisée des contradictions et incohérences temporelles**, et une **synthèse qualitative GraphRAG par communautés de Louvain**.

---

## 1. Vue d'ensemble de l'architecture

Le moteur dépasse les limites des approches purement vectorielles ou par mots-clés en intégrant une modélisation sémantique stricte, une suite de requêtes SPARQL d'audit, et une validation d'intégrité avant calcul du score.

```
src/
├── core/
│   ├── ontology.py          # Namespaces (cv, skill, skos), schéma RDF, taxonomies & distances taxopathiques
│   ├── models.py            # Schémas Pydantic v2 : Entity, Relation, ExtractedGraph, ConflictReport
│   └── validator.py         # Moteur de validation : inversions temporelles, anachronismes, sur-déclarations
├── graph/
│   ├── builder.py           # Ingestion triplets RDFLib & NetworkX (MultiDiGraph) avec réification d'arêtes
│   ├── sparql_queries.py    # Suite de requêtes SPARQL (compétences par rôle, basse confiance, roll-up SKOS)
│   ├── matching.py          # Scoring topologique : Personalized PageRank (PPR), similarité SKOS, Jaccard
│   └── graph_rag.py         # Clustering par communautés Louvain & prompts de synthèse LLM
├── pipeline/
│   ├── extractor.py         # Extraction de relations & qualificatifs temporels (LLM + parser déterministe)
│   └── ranker.py            # Ranker hybride : compute_final_rank(vector, graph, penalty)
├── extraction/              # Extracteur historique regex (rétro-compatibilité)
├── parsing/                 # Parsing de documents PDF (pdfplumber), DOCX (python-docx), TXT
└── scoring/                 # Moteur de scoring de base
tests/
├── test_contradictions.py   # Tests d'inversions, anachronismes, sur-déclarations et ConflictReport
├── test_sparql_queries.py   # Tests des requêtes SPARQL (rôles, seuils de confiance, property paths skos:broader*)
├── test_ranking_pipeline.py # Test synthétique de bout-en-bout (extraction ➔ graphe ➔ scoring ➔ GraphRAG)
├── test_graph_builder.py    # Tests d'ingestion RDFLib, conversion NetworkX et réification d'arêtes
└── test_ranking.py          # Tests de PPR, crédit partiel taxonomique, ranker hybride et critique
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
│    - Relations qualifiées : dates, durées, confidence  │
│    - Snippets sources contextuels (source_snippet)     │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ 2. Validation Pré-Scoring & Détection de Conflits      │
│    - Inversions temporelles (start_date > end_date)     │
│    - Anachronismes (usage techno < date de sortie)      │
│    - Sur-déclaration cumulative (années > contrats)    │
│    ➔ ConflictReport(conflicts, penalty_score)          │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ 3. Représentation Ontologique & Audit SPARQL           │
│    - Schéma RDF/SKOS (cv:, skill:, skos:)              │
│    - Réification des triplets (rdf:Statement)          │
│    - Requêtes SPARQL : rôles, confiance, taxonomie     │
│    - Conversion bidirectionnelle vers nx.MultiDiGraph  │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ 4. Analyse Topologique & Scoring GraphRAG              │
│    - Personalized PageRank (PPR) diffusé depuis l'offre│
│    - Crédit partiel pour compétences proches (SKOS)    │
│    - Détection de communautés (Louvain) & Prompts LLM  │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ 5. Synthèse & Classement Multi-Facteurs                │
│    compute_final_rank(vector, graph, penalty, weights) │
│    FinalScore = α·Vector + β·GraphMatch - γ·Penalty    │
│    Tableau de classement + Audit d'explicabilité       │
└────────────────────────────────────────────────────────┘
```

---

## 3. Formule de Scoring Hybride

Le classement des candidats repose sur la fonction `compute_final_rank` :

$$\text{FinalScore} = \alpha \cdot \text{VectorScore} + \beta \cdot \text{GraphScore} - \gamma \cdot \text{PenaltyScore}$$

Par défaut : $\alpha = 0.4$, $\beta = 0.5$, $\gamma = 0.1$ (ou paramétrables selon les besoins RH).

| Composante | Rôle & Méthode de Calcul |
|---|---|
| **$\text{VectorScore}$** | Similarité textuelle TF-IDF et recouvrement lexical entre le profil et l'offre d'emploi. |
| **$\text{GraphScore}$** | Synthèse topologique combinant :<br>• **Personalized PageRank (PPR)** : diffusion d'importance depuis les compétences requises par l'offre vers le nœud candidat.<br>• **Distance Taxopathique SKOS** : attribution d'un crédit partiel pour les technologies parentes ou connexes (ex. *PyTorch* ➔ *Deep Learning* = 1 saut, *TensorFlow* = 2 sauts).<br>• **Similarité de Jaccard** : taux de recouvrement strict des compétences. |
| **$\text{PenaltyScore}$** | Pénalité déductive issue du `ConflictReport` proportionnelle à la sévérité des anomalies détectées. |

---

## 4. Suite de Requêtes SPARQL d'Audit (`sparql_queries.py`)

Trois requêtes SPARQL modulaires permettent d'interroger et d'auditer le graphe RDF en profondeur :

1. **`get_skills_for_role(graph, role_regex)`** :
   Traverse les propriétés `cv:hasExperience` ➔ `cv:usesSkill` (et `cv:heldRole`) avec filtre Regex insensible à la casse sur le libellé du poste (ex. `'Machine Learning'`).
2. **`get_low_confidence_relations(graph, threshold=0.70)`** :
   Interroge les réifications `rdf:Statement` dont la propriété `cv:confidence < threshold` pour isoler les ambiguïtés d'extraction et inspecter `cv:sourceSnippet`.
3. **`match_skills_with_taxonomy(graph, target_parent_skill)`** :
   Exploite les chemins de propriétés transitifs SPARQL (`skos:broader*`) pour identifier l'ensemble des compétences feuilles du candidat rattachées à une catégorie parente (ex. trouver *PyTorch* et *TensorFlow* via la recherche de `'machinelearning'`).

---

## 5. Détection Automatisée des Incohérences (`validator.py`)

Avant toute étape de scoring, le module `ConsistencyValidator` applique des règles de cohérence et retourne un `ConflictReport` :
1. **Inversions temporelles** : détection des relations où la date de début est postérieure à la date de fin (`start_date > end_date`).
2. **Anachronismes technologiques** : vérification de la date de première utilisation déclarée par rapport au registre officiel des versions (FastAPI : 2018, PyTorch : 2016, Docker : 2013, Kubernetes : 2014, etc.).
3. **Sur-déclaration cumulative** : union des intervalles d'emplois documentés pour vérifier si le volume d'années déclaré sur une compétence est effectivement étayé par les postes occupés.
4. **Vérification d'hallucination** : contrôle des entités extraites par rapport aux limites textuelles du document source.

---

## 6. Synthèse GraphRAG & Détection de Communautés (`graph_rag.py`)

Le module `GraphRAGSummaryEngine` :
- Segmente le sous-graphe du candidat en modules cohérents grâce à l'algorithme de **modularité de Louvain** (`networkx.algorithms.community.louvain_communities`).
- Génère des entrées de prompts structurés par communauté via `generate_community_prompts` (ex. *"Cluster 1: Embedded IoT & Hardware"*, *"Cluster 2: Machine Learning"*) pour alimenter une synthèse LLM des forces et lacunes.
- Produit un audit qualitatif explicable (`generate_candidate_critique`) détaillant :
  - Les compétences validées directement vs déduites par ontologie.
  - La continuité de la trajectoire professionnelle.
  - Le résumé exhaustif des alertes temporelles et de cohérence.

---

## 7. Installation & Prérequis

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

## 8. Exécution des Tests

La suite de tests unitaires valide l'ensemble du pipeline (détection d'incohérences, requêtes SPARQL, pipeline synthétique de ranking, graphe RDFLib/NetworkX et GraphRAG) :

```bash
python -m pytest tests/test_contradictions.py tests/test_sparql_queries.py tests/test_ranking_pipeline.py -v
```

Exécution de l'ensemble des 20 tests du projet :
```bash
python -m pytest tests/test_contradictions.py tests/test_sparql_queries.py tests/test_ranking_pipeline.py tests/test_graph_builder.py tests/test_ranking.py -v
```

Exemple de sortie :
```
tests/test_contradictions.py::test_clean_candidate_passes_validation PASSED
tests/test_contradictions.py::test_temporal_inversion_detected PASSED
tests/test_contradictions.py::test_anachronism_detected PASSED
tests/test_contradictions.py::test_cumulative_overclaiming_detected PASSED
tests/test_contradictions.py::test_multiple_contradictions_accumulate_penalty PASSED
tests/test_contradictions.py::test_validate_conflicts_returns_conflict_report PASSED
tests/test_sparql_queries.py::test_get_skills_for_role PASSED
tests/test_sparql_queries.py::test_get_low_confidence_relations PASSED
tests/test_sparql_queries.py::test_match_skills_with_taxonomy PASSED
tests/test_ranking_pipeline.py::test_end_to_end_synthetic_ranking_pipeline PASSED
tests/test_ranking_pipeline.py::test_ranking_penalizes_contradictions_with_compute_final_rank PASSED
tests/test_graph_builder.py::test_payload_to_rdflib_ingestion PASSED
tests/test_graph_builder.py::test_rdflib_to_networkx_preserves_attributes PASSED
tests/test_graph_builder.py::test_bidirectional_conversion_networkx_to_rdflib PASSED
tests/test_ranking.py::test_ontological_distance_and_partial_credit PASSED
tests/test_ranking.py::test_clean_candidate_ranking_flow PASSED
tests/test_ranking.py::test_contradictory_candidate_receives_penalty PASSED
tests/test_ranking.py::test_ppr_diffusion_and_community_critique PASSED
tests/test_ranking.py::test_extractor_end_to_end_parsing PASSED
tests/test_ranking.py::test_multiple_candidate_ranking_order PASSED

============================= 20 passed in 3.05s ==============================
```

---

## 9. Lancer l'Interface Streamlit

```bash
streamlit run app.py
```
Accédez ensuite à `http://localhost:8501` pour tester l'application interactive.

---

## 10. Exemple d'Utilisation en Python

```python
from src.pipeline.extractor import KnowledgeGraphExtractor
from src.pipeline.ranker import HybridRanker, compute_final_rank
from src.graph.builder import GraphBuilder
from src.graph.sparql_queries import get_skills_for_role, get_low_confidence_relations, match_skills_with_taxonomy
from src.graph.graph_rag import GraphRAGSummaryEngine

# 1. Extraction des graphes candidat et offre
extractor = KnowledgeGraphExtractor(use_llm=False)
cv_graph = extractor.extract_candidate(cv_text, candidate_id="cand_1", candidate_name="Dr. Geoffrey Hinton")
jd_graph = extractor.extract_job(job_text, job_id="job_1", title="Lead AI Engineer")

# 2. Ingestion RDFLib et audit SPARQL
builder = GraphBuilder(include_ontology_taxonomies=True)
rdf_graph, nx_graph = builder.build_candidate_graph(cv_graph)

# Requête SPARQL : compétences utilisées pour un rôle donné
ml_skills = get_skills_for_role(rdf_graph, role_regex="Scientist")

# Requête SPARQL : extraction d'ambiguïtés sous un seuil de confiance
low_conf = get_low_confidence_relations(rdf_graph, threshold=0.70)

# Requête SPARQL : roll-up taxonomique transitif (skos:broader*)
parent_matches = match_skills_with_taxonomy(rdf_graph, target_parent_skill="machinelearning")

# 3. Calcul du score final avec compute_final_rank
ranker = HybridRanker()
result = ranker.score_candidate(cv_graph, jd_graph)
final_score = compute_final_rank(
    vector_score=result.vector_similarity,
    graph_score=result.graph_match_score,
    penalty_score=result.conflict_penalty,
    weights=(0.4, 0.5, 0.1),
)

print(f"Score final : {final_score:.4f}")
print(f"PPR Diffusion : {result.graph_breakdown.ppr_score}")

# 4. Synthèse GraphRAG par communautés Louvain
engine = GraphRAGSummaryEngine(use_llm=False)
prompts = engine.generate_community_prompts(cv_graph)
critique = engine.generate_candidate_critique(cv_graph, jd_graph)
print(critique)
```
