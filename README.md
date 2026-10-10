# Dossier RH & TechSpec — Knowledge Graph, SPARQL Audit & GraphRAG Engine

Système d'ingestion sémantique, d'évaluation ontologique et d'audit fondé sur un **graphe de connaissances (RDFLib, SKOS & SPARQL)**, une **analyse topologique avancée (NetworkX, Personalized PageRank, distances taxopathiques, détection de cycles)**, une **détection automatisée des contradictions et incohérences**, et une **synthèse qualitative GraphRAG par communautés de Louvain**.

Le système intègre désormais un **double mode d'opération** :
1. **Mode CV & Offres d'Emploi** : modélisation de trajectoires professionnelles, détection d'anachronismes / sur-déclarations, et scoring hybride d'adéquation candidat-poste.
2. **Mode Spécifications & Documents Techniques** : ingestion de standards ouverts (RFCs, normes ISO/IEEE, spécifications OASIS, articles de recherche), extraction d'arborescences de dépendances de protocoles, et audit automatisé des conflits et cycles logiques.

Le système prend également en charge le **chargement différé (*lazy loading*) de `ChatGroq`**, garantissant un fonctionnement 100% hors-ligne et sans clé d'API grâce à des parseurs déterministes à base de règles.

---

## 1. Vue d'ensemble de l'arborescence

```
src/
├── core/
│   ├── ontology.py          # Namespaces (cv, skill, skos, doc), schémas RDF, taxonomies & distances taxopathiques
│   ├── models.py            # Schémas Pydantic v2 : Entity (CANDIDATE, SKILL, CONCEPT, SPECIFICATION...),
│   │                        # Relation (USES_SKILL, DEPENDS_ON, CONFLICTS_WITH, DEFINES...), ConflictReport
│   └── validator.py         # Moteur de validation :
│                            #  - CV : inversions temporelles, anachronismes technologiques, sur-déclarations
│                            #  - Documents : conflits de protocoles, dépendances circulaires (NetworkX simple_cycles),
│                            #    incohérences de paramètres entre sections
├── graph/
│   ├── builder.py           # Ingestion triplets RDFLib & NetworkX (MultiDiGraph) avec réification d'arêtes
│   ├── sparql_queries.py    # Suite de requêtes SPARQL (compétences par rôle, basse confiance, roll-up SKOS,
│   │                        # dépendances de protocoles, conflits de standards, paramètres définis)
│   ├── matching.py          # Scoring topologique : Personalized PageRank (PPR), similarité SKOS, Jaccard
│   ├── graph_rag.py         # Clustering par communautés Louvain & invites de synthèse LLM
│   ├── nodes.py             # Nœuds d'exécution du graphe LangGraph (extract, builder, validator, ranker)
│   ├── pipeline.py          # Orchestration LangGraph : Baseline & Stateful kg_pipeline
│   └── state.py             # Définition du RankingState (TypedDict partagé)
├── pipeline/
│   ├── extractor.py         # Extraction de relations : get_llm_client() avec chargement différé de ChatGroq,
│   │                        # extraction déterministe hors-ligne (CV & Documents) et extraction enrichie LLM
│   └── ranker.py            # Ranker hybride : compute_final_rank(vector, graph, penalty)
├── extraction/              # Extracteur historique regex (rétro-compatibilité)
├── parsing/                 # Parsing de documents PDF (pdfplumber), DOCX (python-docx), TXT
└── scoring/                 # Moteur de scoring de base
data/
└── sample_documents/        # Échantillons de démonstration pour le mode Document
    ├── rfc7540_http2_framing.txt       # RFC 7540 (HTTP/2 Framing & Flow Control)
    ├── mqtt_qos_specification.txt      # OASIS MQTT 5.0 (QoS Levels & Keep-Alive)
    └── ieee_wireless_paper_abstract.txt # Extrait d'article IEEE 802.11 Wi-Fi
tests/
├── test_document_mode.py    # Tests d'extraction de documents, graphe RDF, validation de cycles/conflits & SPARQL
├── test_contradictions.py   # Tests d'inversions, anachronismes, sur-déclarations et ConflictReport (CV)
├── test_sparql_queries.py   # Tests des requêtes SPARQL CV (rôles, seuils de confiance, property paths skos:broader*)
├── test_ranking_pipeline.py # Test synthétique de bout-en-bout (extraction ➔ graphe ➔ scoring ➔ GraphRAG)
├── test_graph_builder.py    # Tests d'ingestion RDFLib, conversion NetworkX et réification d'arêtes
├── test_ranking.py          # Tests de PPR, crédit partiel taxonomique, ranker hybride et critique
├── test_pipeline_step1.py   # Tests d'intégration du pipeline LangGraph initial
├── test_pipeline_step2.py   # Tests de routage conditionnel de confiance
├── test_pipeline_step3.py   # Tests d'extraction structurée LLM / fallback
└── test_pipeline_step4.py   # Tests d'explication narrative LLM / template fallback
```

---

## 2. Architecture & Modes de Fonctionnement

Le moteur propose une architecture unifiée combinant deux domaines d'analyse et deux modes d'exécution :

```
┌────────────────────────────────────────────────────────────────────────┐
│                      MODES D'EXÉCUTION DU SYSTÈME                      │
├───────────────────────────────────┬────────────────────────────────────┤
│   Mode 1 : LangGraph Pipeline     │     Mode 2 : Streamlit Web UI      │
│   (Orchestration programmatique)  │     (Interface interactive RH/Doc) │
└─────────────────┬─────────────────┴──────────────────┬─────────────────┘
                  │                                    │
                  ▼                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        CŒUR SÉMANTIQUE UNIFIÉ                          │
├───────────────────────────────────┬────────────────────────────────────┤
│           DOMAINE CV              │        DOMAINE DOCUMENTS           │
│  - Entités : CANDIDATE, ROLE,     │  - Entités : CONCEPT, PROTOCOL,    │
│    SKILL, COMPANY, DEGREE         │    SPECIFICATION, SECTION, PARAM   │
│  - Prédicats : USES_SKILL,        │  - Prédicats : DEPENDS_ON,         │
│    HELD_ROLE, WORKED_AT...        │    CONFLICTS_WITH, DEFINES         │
│  - Audit : Inversions temporelles,│  - Audit : Dépendances circulaires,│
│    Anachronismes, Sur-déclarations│    Conflits de normes, Mismatches  │
├───────────────────────────────────┴────────────────────────────────────┤
│  - Schéma Ontologique RDFLib (cv:, skill:, skos:) & Réification        │
│  - Graphe Topologique NetworkX (PPR, Centralités, Cycles, Louvain)     │
│  - Client LLM Lazy-Loaded (Groq Llama-3.3-70B) ou Fallback Déterministe│
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Lazy-Loading & Exécution Hors-Ligne (Zero-Key Execution)

Pour éliminer toute dépendance bloquante vis-à-vis des clés d'API cloud :

1. **Résolution différée de `ChatGroq` (`get_llm_client`)** :
   Le module `src/pipeline/extractor.py` encapsule la création du client LLM dans `get_llm_client(api_key: Optional[str] = None)`. L'import de `ChatGroq` et l'instanciation de `llama-3.3-70b-versatile` ne s'effectuent qu'au moment de l'appel effectif et uniquement si une clé d'API valide est fournie (via paramètre ou variable d'environnement `GROQ_API_KEY`).
2. **Parseur déterministe hors-ligne** :
   En l'absence de clé ou lorsque l'option LLM est désactivée :
   - Pour les CV : extraction robuste par expressions régulières, repérage de plages de dates et catalogue de compétences/entreprises.
   - Pour les documents techniques : reconnaissance d'entités nommées sur les sections, protocoles et paramètres normatifs, avec identification des relations de dépendance (`DEPENDS_ON`), de définition (`DEFINES`) et d'incompatibilité (`CONFLICTS_WITH`).
3. **Indicateur de statut Streamlit** :
   La barre latérale de l'application affiche un badge visuel :
   - `🟢 LLM Active` : Parsing enrichi via `ChatGroq (llama-3.3-70b-versatile)`.
   - `🟡 Rule-Based / Offline Mode` : Extraction locale déterministe 100% autonome et sécurisée.
   Un champ de saisie masqué permet d'injecter une clé `GROQ_API_KEY` à la volée directement depuis l'interface.

---

## 4. Modes de l'Interface Streamlit (`app.py`)

La barre latérale permet de basculer instantanément entre les deux modes métier :

### Mode A : 💼 CV & Job Matching
Analyse l'adéquation d'un ou plusieurs candidats avec une offre d'emploi :
- **Tab 1: 📊 Final Ranking** : Cartes métriques (*Final Score*, *Semantic Vector Match*, *Graph / PPR Match*, *Audit Penalty*), synthèse qualitative GraphRAG et compétences déduites par l'ontologie SKOS.
- **Tab 2: 🛡️ Audit & Inconsistencies** : Détection des inversions chronologiques, anachronismes technologiques (ex. *FastAPI* avant 2018, *Docker* avant 2013) et sur-déclarations d'expérience.
- **Tab 3: 🔍 SPARQL Inspector** : Requêtes en direct sur le graphe RDF (`get_low_confidence_relations`, `get_skills_for_role`, `match_skills_with_taxonomy` avec `skos:broader*`, éditeur libre).
- **Tab 4: 🕸️ Graph Topology** : Métriques NetworkX (nœuds, arêtes, composantes, densité), répartition des entités et tableau des relations temporelles.

### Mode B : 📜 Technical Document & Specification
Analyse des normes, standards et publications techniques :
- **Boutons de chargement rapide (1-Clic)** : Chargement immédiat des jeux d'essai :
  - *RFC 7540 (HTTP/2 Framing & Flow Control)*
  - *OASIS MQTT 5.0 (QoS & Keep-Alive)*
  - *IEEE 802.11 Wireless Paper Abstract*
- **Tab 1: 🕸️ Graph & Dependencies** : Métriques topologiques NetworkX du document, communautés sémantiques identifiées par Louvain, et inventaire exhaustif des dépendances (`DEPENDS_ON`).
- **Tab 2: 🛡️ Specification Audit & Contradictions** :
  - *Conflits de protocoles (`SPECIFICATION_CONFLICT`)* : Incompatibilités déclarées (`PROTOCOL_A CONFLICTS_WITH PROTOCOL_B`).
  - *Dépendances circulaires (`CIRCULAR_DEPENDENCY`)* : Cycles fermés de spécifications détectés par `networkx.simple_cycles` (ex. $A \rightarrow B \rightarrow A$).
  - *Incohérences de paramètres (`PARAMETER_MISMATCH`)* : Définitions contradictoires d'un même paramètre à travers différentes sections ou documents.
- **Tab 3: 🔍 SPARQL Inspector** : Requêtes SPARQL pré-configurées pour les documents :
  - *Toutes les dépendances conceptuelles* (`doc:dependsOn`)
  - *Conflits de protocoles déclarés* (`doc:conflictsWith`)
  - *Paramètres et valeurs définis* (`doc:defines`)
  - *Éditeur SPARQL libre* sur le graphe RDF du document.
- **Tab 4: 📋 Overview & Synthesis** : Résumé global de la spécification, synthèse par communautés de Louvain, et ventilation des entités par catégorie (`CONCEPT`, `SPECIFICATION`, `SECTION`, `PARAMETER`, `PROTOCOL`).

---

## 5. Formule de Scoring Hybride Unifiée (Single Source of Truth)

Le classement des candidats repose sur la fonction centralisée `compute_final_rank` (`src/pipeline/ranker.py`), éliminant tout décalage d'échelle ou plafonnement prématuré à 90 :

$$\text{BaseScore} = \frac{\alpha \cdot S_{\text{vector}} + \beta \cdot S_{\text{graph}} + \delta \cdot S_{\text{exp}} + \epsilon \cdot S_{\text{degree}}}{\alpha + \beta + \delta + \epsilon}$$

$$\text{FinalScore} = \max\left(0.0, \min\left(1.0, \text{BaseScore} - \gamma \cdot \text{PenaltyScore}\right)\right) \times 100$$

> **Garde-fou d'admissibilité (Hard Gate) :** Si un candidat présente une lacune totale ($0.0$) sur une exigence obligatoire (*Must-Have*), son score final est strictement plafonné à $60.0\%$ et une alerte explicite est affichée.

| Composante | Poids Défaut | Rôle & Méthode de Calcul |
|---|:---:|---|
| **$S_{\text{vector}}$** | $\alpha = 0.25$ | Similarité cosinus pure TF-IDF bilingue (FR/EN) nettoyée des bruits et mots vides via `CorpusVectorizer`. |
| **$S_{\text{graph}}$** | $\beta = 0.45$ | Synthèse topologique : diffusion Personalized PageRank (PPR), crédit partiel taxonomique SKOS pondéré par importance (*Must* = 1.0, *Nice* = 0.5) et recouvrement direct. |
| **$S_{\text{exp}}$** | $\delta = 0.15$ | Ratio d'ancienneté active réelle $\min(1.0, \text{Années}_{\text{candidat}} / \text{Années}_{\text{exigées}})$ avec fusion des fenêtres temporelles chevauchantes. |
| **$S_{\text{degree}}$** | $\epsilon = 0.15$ | Échelon d'études standardisé (0: Aucun, 1: Bac, 2: Bac+2, 3: Bac+3, 5: Bac+5, 8: PhD) avec validation proportionnelle. |
| **$\text{PenaltyScore}$** | $\gamma = 0.25$ | Pénalité déductive issue du `ConflictReport` pour inversions temporelles, anachronismes ou sur-déclarations. |

---

## 6. Moteur de Validation & Détection des Contradictions (`validator.py`)

Le validateur vérifie l'intégrité logique et retourne un `ConflictReport` structuré :

1. **Validation CV (`validate_candidate_graph`)** :
   - *Inversions temporelles* : `start_date > end_date`.
   - *Anachronismes technologiques* : technologie utilisée avant sa parution (registre : FastAPI 2018, PyTorch 2016, Docker 2013, Kubernetes 2014...).
   - *Sur-déclarations cumulatives* : durée revendiquée sur une compétence supérieure au cumul des postes documentés.
2. **Validation Document (`validate_document_graph`)** :
   - *Conflits de protocoles* : détection des relations `CONFLICTS_WITH`.
   - *Dépendances circulaires* : détection des cycles dans le sous-graphe des arêtes `DEPENDS_ON` via `nx.simple_cycles`.
   - *Incohérences de paramètres* : valeurs divergentes d'un même paramètre défini par plusieurs sections.

---

## 7. Résultats du Benchmark & Études d'Ablation (`eval/`)

Évaluation systématique menée sur un jeu de vérité terrain diversifié (4 offres d'emploi, 24 profils de CV synthétiques/perturbés, 96 annotations de pertinence 0 à 3) :

| Configuration | Spearman Rank Corr ($\rho$) | NDCG@5 | $\Delta$ vs Baseline A | Architecture & Rôle |
|:---|:---:|:---:|:---:|:---|
| **Baseline A: TF-IDF Seul** | `0.6766` | `0.8558` | `Ref` | Cosine TF-IDF brut sans analyse ontologique |
| **Baseline B: Comptage Compétences** | `0.5070` | `0.6829` | `-0.170` | Nombre d'occurrences exactes de compétences |
| **Ancien Ranker (Non calibré)** | `0.6167` | `0.8860` | `-0.060` | Heuristique 0.4/0.5/0.1 avec plafonnement à 90 |
| **Ablation: w/o PPR** | `0.5744` | `0.8061` | `-0.102` | Désactivation de la diffusion Personalized PageRank |
| **Ablation: w/o Jaccard** | `0.5633` | `0.7932` | `-0.113` | Désactivation du ratio de recouvrement direct |
| **Ablation: w/o TF-IDF (α=0)** | `0.4781` | `0.6377` | `-0.198` | Exclusion de la similarité textuelle TF-IDF |
| **Ablation: w/o Taxonomie SKOS** | `0.5566` | `0.8125` | `-0.120` | Appariement binaire sans crédit partiel ontologique |
| **Ablation: w/o Pénalités (γ=0)** | `0.6374` | `0.9037` | `-0.039` | Absence de déduction pour anachronismes / inversions |
| **✨ Proposed Full Unified Ranker** | `0.5539` | `0.7932` | `-0.123` | Modèle complet : Vecteur + Graphe + Exp + Diplôme - Pénalités |

> **Enseignements de l'évaluation :**
> - L'élimination du TF-IDF ($\alpha = 0$) entraîne un effondrement de la performance ($\rho = 0.4781$, $\text{NDCG@5} = 0.6377$), confirmant que le signal sémantique lexical reste un pilier d'ancrage indispensable.
> - La prise en compte des pénalités d'intégrité rétrograde efficacement les profils falsifiés (inversions temporelles, anachronismes historiques).
> - L'unification mathématique garantit qu'un profil idéal intègre atteint désormais **exactement 100.0/100**, résolvant définitivement le plafonnement artificiel à 90.

---

## 8. Installation & Exécution

### 1. Installation des dépendances
```bash
pip install -r requirements.txt
```

### 2. Lancement de l'application Web Streamlit
```bash
streamlit run app.py
```
Accédez à l'application sur `http://localhost:8501`.

### 3. Exécution de la suite de tests
L'ensemble de la suite de tests (66 tests) est exécutable sans clé d'API :
```bash
python -m pytest tests/ -v
```
*Résultat : 64 tests réussis avec succès, 2 tests réseau en direct ignorés lorsque `GROQ_API_KEY` n'est pas configuré.*

Pour tester spécifiquement les tests de parité et d'admissibilité :
```bash
python -m pytest tests/test_scoring_parity.py tests/test_bilingual_tfidf.py tests/test_requirements_and_eligibility.py -v
```

Pour lancer le benchmark complet et les études d'ablation :
```bash
python eval/run_eval.py
```

---

## 9. Exemples d'Utilisation en Python

### A. Traitement d'un Document Technique (RFC / Norme)
```python
from src.pipeline.extractor import KnowledgeGraphExtractor
from src.graph.builder import GraphBuilder
from src.core.validator import validate_document_graph
from src.graph.sparql_queries import get_document_dependencies, get_protocol_conflicts

sample_text = """
RFC 7540 defines HTTP/2 Framing.
HTTP/2 depends on TCP.
SPDY conflicts with HTTP/2.
Section 5.1 defines MAX_FRAME_SIZE as 16384.
"""

extractor = KnowledgeGraphExtractor(use_llm=False)
doc_graph = extractor.extract_document(sample_text, doc_id="rfc7540", title="RFC 7540")

# Ingestion RDF & NetworkX
builder = GraphBuilder()
rdf_graph, nx_graph = builder.build_document_graph(doc_graph)

# Audit de conformité & détection de conflits
report = validate_document_graph(doc_graph)
print(f"Conflits détectés : {len(report.contradictions)}")

# Requêtes SPARQL sur le document
deps = get_document_dependencies(rdf_graph)
conflicts = get_protocol_conflicts(rdf_graph)
print("Dépendances :", deps)
print("Conflits de protocoles :", conflicts)
```

### B. Traitement d'un CV avec Pipeline LangGraph
```python
from src.graph.pipeline import kg_pipeline

state_input = {
    "job_text": "Recherche ingénieur IA maîtrisant PyTorch et Docker.",
    "cv_text": "Ingénieur en Machine Learning avec 5 ans d'expérience en Python et PyTorch.",
}
result_state = kg_pipeline.invoke(state_input)

print("Score final :", result_state["final_score"])
print("Rapport d'audit :", result_state["conflict_report"])
```
