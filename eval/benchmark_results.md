### 📊 Benchmark & Ablation Study Results (Ground Truth: 4 Jobs, 24 CVs, 96 Ratings)

| Configuration | Spearman Rank Corr (rho) | NDCG@5 | Delta vs Baseline A | Architecture & Rôle |
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
