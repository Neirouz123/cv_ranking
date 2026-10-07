# Dossier RH — Knowledge Graph, SPARQL Audit & GraphRAG CV Ranking Engine

Système d'évaluation et de classement de candidats fondé sur un **graphe de connaissances ontologique (RDFLib, SKOS & SPARQL)**, une **analyse topologique avancée (NetworkX, Personalized PageRank, distances taxopathiques)**, une **détection automatisée des contradictions et incohérences temporelles**, et une **synthèse qualitative GraphRAG par communautés de Louvain**.

---

## 1. Vue d'ensemble de l'arborescence

Le moteur dépasse les limites des approches purement vectorielles ou par mots-clés en combinant une modélisation sémantique stricte, une suite de requêtes SPARQL d'audit, une validation d'intégrité avant calcul du score, et une orchestration par graphe d'états (LangGraph) :

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
│   ├── graph_rag.py         # Clustering par communautés Louvain & prompts de synthèse LLM
│   ├── nodes.py             # Nœuds d'exécution du graphe LangGraph (extract, builder, validator, ranker)
│   ├── pipeline.py          # Orchestration LangGraph : Baseline & Stateful kg_pipeline
│   └── state.py             # Définition du RankingState (TypedDict partagé)
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
├── test_ranking.py          # Tests de PPR, crédit partiel taxonomique, ranker hybride et critique
├── test_pipeline_step1.py   # Tests d'intégration du pipeline LangGraph initial
├── test_pipeline_step2.py   # Tests de routage conditionnel de confiance
├── test_pipeline_step3.py   # Tests d'extraction structurée LLM / fallback
└── test_pipeline_step4.py   # Tests d'explication narrative LLM / template fallback
```

---

## 2. Architecture & Execution Modes

Le système dispose d'une architecture unifiée offrant deux modes d'exécution complémentaires partageant le même socle sémantique, ontologique et topologique :

```
┌────────────────────────────────────────────────────────────────────────┐
│                      MODES D'EXÉCUTION DU SYSTÈME                      │
├───────────────────────────────────┬────────────────────────────────────┤
│   Mode 1 : LangGraph Pipeline     │     Mode 2 : Streamlit Web UI      │
│   (Orchestration programmatique)  │     (Interface RH interactive)     │
└─────────────────┬─────────────────┴──────────────────┬─────────────────┘
                  │                                    │
                  ▼                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        CŒUR SÉMANTIQUE UNIFIÉ                          │
│  - Extraction Pydantic v2 (Entités, Relations temporelles, Snippets)   │
│  - Schéma Ontologique RDFLib (cv:, skill:, skos:) & Réification        │
│  - Graphe Topologique NetworkX (Personalized PageRank, Centralités)    │
│  - Audit d'Intégrité (Inversions, Anachronismes, Sur-déclarations)     │
│  - Synthèse GraphRAG par communautés de Louvain                        │
└────────────────────────────────────────────────────────────────────────┘
```

### A. LangGraph Pipeline (Stateful Orchestration Engine)

Pour les flux automatisés, les traitements par lots ou l'intégration API, le pipeline d'états `kg_pipeline` (`src/graph/pipeline.py` & `src/graph/nodes.py`) orchestre l'exécution séquentielle et contrôlée de chaque étape via un graphe d'états typé (`RankingState`) :

```
START ──► extract_node ──► graph_construction_node ──► validation_node ──► ranking_node ──► END
```

| Nœud LangGraph | Description & Rôle dans le flux |
|---|---|
| **`extract_node`** | Extrait les entités (`CANDIDATE`, `ROLE`, `SKILL`, `COMPANY`, `DEGREE`) et les relations temporelles qualifiées (`start_date`, `end_date`, `confidence`, `source_snippet`) depuis le texte brut de l'offre et du CV pour construire les objets `job_graph` et `candidate_graph` (`ExtractedGraph`). |
| **`graph_construction_node`** | Ingeste le graphe candidat dans un graphe sémantique **RDFLib** appliquant les namespaces ontologiques (`cv:`, `skill:`, `skos:`) et réifiant les relations avec niveau de confiance, puis convertit ce graphe vers un **`nx.MultiDiGraph`** NetworkX préservant tous les attributs d'arêtes. |
| **`validation_node`** | Exécute le moteur d'intégrité `validate_candidate_graph` pour détecter les inversions chronologiques, les anachronismes technologiques et la sur-déclaration de compétences, et produit un `ConflictReport` assorti d'un coefficient de pénalité proportionnel. |
| **`ranking_node`** | Calcule le score hybride final via `compute_final_rank(vector_score, graph_score, penalty_score)`, évalue la diffusion Personalized PageRank (PPR), applique le crédit partiel SKOS, et génère la critique qualitative GraphRAG par communautés de Louvain. |

### B. Streamlit Web UI (`app.py`)

L'interface web interactive `app.py` fournit aux recruteurs et auditeurs RH un tableau de bord complet avec une inspection granulaire répartie en **4 onglets spécialisés** par candidat :

#### 📊 Tab 1: Final Ranking
- **Cartes métriques de synthèse** :
  - **Final Score** : Score global normalisé sur 100.
  - **Semantic Vector Match** : Similarité lexicale / sémantique entre le CV et l'offre d'emploi.
  - **Graph / PPR Match** : Score topologique issu de la diffusion Personalized PageRank et des distances ontologiques SKOS.
  - **Audit Penalty** : Déduction chiffrée issue des anomalies de cohérence détectées.
- **Synthèse qualitative GraphRAG** : Diagnostic textuel expliquant l'adéquation du profil, la logique de parcours et les atouts majeurs.
- **Détail des compétences appariées** :
  - *Correspondances directes* : Compétences exigées retrouvées explicitement dans le profil.
  - *Compétences déduites (Ontologie SKOS)* : Équivalences identifiées par taxonomie avec nombre de sauts de distance et score partiel associé (ex. *PyTorch* ➔ *Deep Learning*).

#### 🛡️ Tab 2: Audit & Inconsistencies
- Affiche l'audit d'intégrité issu de `validate_candidate_graph`.
- Si le profil est irréprochable : badge vert signalant un profil intègre et vérifié.
- En présence d'incohérences : cartes d'alerte détaillées classées par sévérité :
  - **Inversions temporelles** : Contrats ou diplômes où `start_date > end_date`.
  - **Anachronismes technologiques** : Détection de l'utilisation déclarée d'une technologie avant sa date officielle de parution (ex. FastAPI avant 2018, PyTorch avant 2016, Docker avant 2013).
  - **Sur-déclarations cumulatives** : Durée totale d'expérience revendiquée sur une compétence supérieure à la durée réelle des postes documentés.

#### 🔍 Tab 3: SPARQL Inspector
Console d'interrogation en temps réel sur le graphe RDF en mémoire du candidat (`rdflib.Graph`) :
- **Contrôle de confiance d'extraction** : Isole automatiquement les triplets reifiés (`rdf:Statement`) présentant un score de confiance inférieur au seuil choisi (`cv:confidence < seuil`, par défaut 0.70) et affiche le snippet source pour vérification humaine.
- **Compétences liées à un rôle spécifique** : Requête paramétrée avec filtre Regex sur l'intitulé de poste pour lister les technologies exploitées sur une expérience cible.
- **Expansion taxonomique transitive (`skos:broader*`)** : Remonte l'ensemble des compétences feuilles du candidat rattachées à une catégorie parente (ex: `machinelearning`, `deeplearning`, `backenddevelopment`).
- **Éditeur SPARQL libre** : Permet à un auditeur d'exécuter n'importe quelle requête SPARQL personnalisée et d'en visualiser instantanément les résultats dans un tableau interactif.

#### 🕸️ Tab 4: Graph Topology
- **Indicateurs structurels NetworkX** : Nombre de nœuds (entités), nombre d'arêtes (relations), composantes faiblement connexes (*weakly connected components*) et densité topologique du sous-graphe candidat.
- **Distribution des entités** : Diagramme en barres représentant la répartition par catégorie (`CANDIDATE`, `ROLE`, `SKILL`, `COMPANY`, `DEGREE`).
- **Tableau exhaustif des relations** : Liste complète des arêtes modélisées avec entité source, prédicat (`HELD_ROLE`, `WORKED_AT`, `USES_SKILL`, `EARNED_DEGREE`), cible, dates et confiance.

### C. Setup & Run

#### 1. Installation des dépendances
Assurez-vous de disposer de Python $\ge 3.10$ et installez les paquets requis :
```bash
pip install -r requirements.txt
```

#### 2. Lancement de l'application Web Streamlit
```bash
streamlit run app.py
```
Ouvrez ensuite votre navigateur sur `http://localhost:8501`.

#### 3. Exécution des tests unitaires
Vérifiez l'intégrité de la suite complète de 45 tests avec :
```bash
python -m pytest tests/ -v
```

---

## 3. Formule de Scoring Hybride

Le classement des candidats repose sur la fonction `compute_final_rank` :

$$\text{FinalScore} = \alpha \cdot \text{VectorScore} + \beta \cdot \text{GraphScore} - \gamma \cdot \text{PenaltyScore}$$

Par défaut : $\alpha = 0.4$, $\beta = 0.5$, $\gamma = 0.1$ (ajustables dynamiquement dans la barre latérale de l'interface Streamlit).

| Composante | Rôle & Méthode de Calcul |
|---|---|
| **$\text{VectorScore}$** | Similarité textuelle TF-IDF et recouvrement lexical entre le profil et l'offre d'emploi. |
| **$\text{GraphScore}$** | Synthèse topologique combinant :<br>• **Personalized PageRank (PPR)** : diffusion d'importance depuis les compétences requises par l'offre vers le nœud candidat.<br>• **Distance Taxopathique SKOS** : attribution d'un crédit partiel pour les technologies parentes ou connexes (ex. *PyTorch* ➔ *Deep Learning* = 1 saut, *TensorFlow* = 2 sauts).<br>• **Similarité de Jaccard** : taux de recouvrement strict des compétences. |
| **$\text{PenaltyScore}$** | Pénalité déductive issue du `ConflictReport` proportionnelle au nombre et à la sévérité des anomalies détectées. |

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

## 7. Exemple d'Utilisation en Python

```python
from src.pipeline.extractor import KnowledgeGraphExtractor
from src.pipeline.ranker import HybridRanker, compute_final_rank
from src.graph.builder import GraphBuilder
from src.graph.sparql_queries import get_skills_for_role, get_low_confidence_relations, match_skills_with_taxonomy
from src.graph.graph_rag import GraphRAGSummaryEngine
from src.graph.pipeline import kg_pipeline

# 1. Utilisation directe via le pipeline LangGraph
state_input = {
    "job_text": "Recherche ingénieur IA maîtrisant PyTorch et Docker.",
    "cv_text": "Ingénieur en Machine Learning avec 5 ans d'expérience en Python et PyTorch.",
}
result_state = kg_pipeline.invoke(state_input)
print("Score hybride LangGraph :", result_state["final_score"])
print("Rapport de conflit :", result_state["conflict_report"])

# 2. Utilisation programmatique modulaire
extractor = KnowledgeGraphExtractor(use_llm=False)
cv_graph = extractor.extract_candidate(state_input["cv_text"], candidate_id="cand_1", candidate_name="Alice")
jd_graph = extractor.extract_job(state_input["job_text"], job_id="job_1", title="Ingénieur IA")

builder = GraphBuilder(include_ontology_taxonomies=True)
rdf_graph, nx_graph = builder.build_candidate_graph(cv_graph)

# Requête SPARQL : extraction d'ambiguïtés sous un seuil de confiance
low_conf = get_low_confidence_relations(rdf_graph, threshold=0.70)

# Requête SPARQL : roll-up taxonomique transitif (skos:broader*)
parent_matches = match_skills_with_taxonomy(rdf_graph, target_parent_skill="machinelearning")

# Calcul du score hybride
ranker = HybridRanker()
result = ranker.score_candidate(cv_graph, jd_graph)
final_score = compute_final_rank(
    vector_score=result.vector_similarity,
    graph_score=result.graph_match_score,
    penalty_score=result.conflict_penalty,
    weights=(0.4, 0.5, 0.1),
)

print(f"Score final : {final_score:.4f}")
```
