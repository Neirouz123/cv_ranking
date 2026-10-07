"""
src/core/ontology.py
--------------------
Ontology definitions using RDFLib and SKOS for CV and Job Description knowledge modeling.
Standardizes prefixes (cv, skill, skos), RDF classes, predicates, and hierarchical taxonomies.
"""

from __future__ import annotations

import re
from typing import Optional
import networkx as nx
from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import OWL, RDF, RDFS, SKOS, XSD

# Standardized namespaces
CV = Namespace("http://recruitment.org/cv#")
SKILL = Namespace("http://recruitment.org/skill#")
# SKOS is standard from rdflib.namespace

# Predefined Skill Hierarchy: (child_id, broader_parent_id, canonical_label)
SKOS_SKILL_RELATIONS: list[tuple[str, str, str]] = [
    # Top-level domains
    ("domain:ai_data", "domain:tech", "AI & Data Science"),
    ("domain:software_engineering", "domain:tech", "Software Engineering"),
    ("domain:devops_cloud", "domain:tech", "Cloud & DevOps"),
    ("domain:data_infrastructure", "domain:tech", "Data Infrastructure"),

    # AI & Data Science
    ("skill:machinelearning", "domain:ai_data", "Machine Learning"),
    ("skill:deeplearning", "skill:machinelearning", "Deep Learning"),
    ("skill:pytorch", "skill:deeplearning", "PyTorch"),
    ("skill:tensorflow", "skill:deeplearning", "TensorFlow"),
    ("skill:keras", "skill:deeplearning", "Keras"),
    ("skill:jax", "skill:deeplearning", "JAX"),

    ("skill:nlp", "skill:machinelearning", "Natural Language Processing"),
    ("skill:transformers", "skill:nlp", "Transformers"),
    ("skill:huggingface", "skill:nlp", "Hugging Face"),
    ("skill:spacy", "skill:nlp", "SpaCy"),
    ("skill:langchain", "skill:nlp", "LangChain"),
    ("skill:graphrag", "skill:nlp", "GraphRAG"),
    ("skill:llm", "skill:nlp", "Large Language Models"),

    ("skill:computervision", "skill:machinelearning", "Computer Vision"),
    ("skill:opencv", "skill:computervision", "OpenCV"),
    ("skill:yolo", "skill:computervision", "YOLO"),

    ("skill:dataanalysis", "domain:ai_data", "Data Analysis"),
    ("skill:scikitlearn", "skill:machinelearning", "Scikit-Learn"),
    ("skill:pandas", "skill:dataanalysis", "Pandas"),
    ("skill:numpy", "skill:dataanalysis", "NumPy"),
    ("skill:scipy", "skill:dataanalysis", "SciPy"),

    # Software Engineering - Backend
    ("skill:backenddevelopment", "domain:software_engineering", "Backend Development"),
    ("skill:python", "skill:backenddevelopment", "Python"),
    ("skill:fastapi", "skill:python", "FastAPI"),
    ("skill:flask", "skill:python", "Flask"),
    ("skill:django", "skill:python", "Django"),

    ("skill:java", "skill:backenddevelopment", "Java"),
    ("skill:springboot", "skill:java", "Spring Boot"),

    ("skill:csharp", "skill:backenddevelopment", "C#"),
    ("skill:dotnet", "skill:csharp", ".NET"),

    ("skill:go", "skill:backenddevelopment", "Go"),
    ("skill:rust", "skill:backenddevelopment", "Rust"),
    ("skill:nodejs", "skill:backenddevelopment", "Node.js"),

    # Software Engineering - Frontend
    ("skill:frontenddevelopment", "domain:software_engineering", "Frontend Development"),
    ("skill:javascript", "skill:frontenddevelopment", "JavaScript"),
    ("skill:typescript", "skill:javascript", "TypeScript"),
    ("skill:react", "skill:frontenddevelopment", "React"),
    ("skill:nextjs", "skill:react", "Next.js"),
    ("skill:vue", "skill:frontenddevelopment", "Vue.js"),
    ("skill:angular", "skill:frontenddevelopment", "Angular"),

    # Cloud & DevOps
    ("skill:clouddevops", "domain:devops_cloud", "Cloud & DevOps"),
    ("skill:docker", "skill:clouddevops", "Docker"),
    ("skill:kubernetes", "skill:clouddevops", "Kubernetes"),
    ("skill:aws", "skill:clouddevops", "AWS"),
    ("skill:gcp", "skill:clouddevops", "GCP"),
    ("skill:azure", "skill:clouddevops", "Azure"),
    ("skill:terraform", "skill:clouddevops", "Terraform"),
    ("skill:cicd", "skill:clouddevops", "CI/CD"),

    # Data Infrastructure
    ("skill:sql", "domain:data_infrastructure", "SQL"),
    ("skill:postgresql", "skill:sql", "PostgreSQL"),
    ("skill:mysql", "skill:sql", "MySQL"),
    ("skill:nosql", "domain:data_infrastructure", "NoSQL"),
    ("skill:mongodb", "skill:nosql", "MongoDB"),
    ("skill:redis", "skill:nosql", "Redis"),
    ("skill:dataengineering", "domain:data_infrastructure", "Data Engineering"),
    ("skill:kafka", "skill:dataengineering", "Kafka"),
    ("skill:spark", "skill:dataengineering", "Apache Spark"),
    ("skill:airflow", "skill:dataengineering", "Apache Airflow"),
    ("skill:snowflake", "skill:dataengineering", "Snowflake"),
]

# Alias and synonym dictionary
SYNONYM_MAP: dict[str, str] = {
    "pytorch": "skill:pytorch",
    "torch": "skill:pytorch",
    "tensorflow": "skill:tensorflow",
    "tf": "skill:tensorflow",
    "keras": "skill:keras",
    "scikit-learn": "skill:scikitlearn",
    "scikitlearn": "skill:scikitlearn",
    "sklearn": "skill:scikitlearn",
    "deep learning": "skill:deeplearning",
    "deeplearning": "skill:deeplearning",
    "machine learning": "skill:machinelearning",
    "machinelearning": "skill:machinelearning",
    "ml": "skill:machinelearning",
    "nlp": "skill:nlp",
    "natural language processing": "skill:nlp",
    "computer vision": "skill:computervision",
    "cv": "skill:computervision",
    "transformers": "skill:transformers",
    "huggingface": "skill:huggingface",
    "hugging face": "skill:huggingface",
    "langchain": "skill:langchain",
    "graphrag": "skill:graphrag",
    "llm": "skill:llm",
    "llms": "skill:llm",
    "pandas": "skill:pandas",
    "numpy": "skill:numpy",
    "fastapi": "skill:fastapi",
    "flask": "skill:flask",
    "django": "skill:django",
    "python": "skill:python",
    "python3": "skill:python",
    "spring": "skill:springboot",
    "spring boot": "skill:springboot",
    "springboot": "skill:springboot",
    "java": "skill:java",
    "javascript": "skill:javascript",
    "js": "skill:javascript",
    "typescript": "skill:typescript",
    "ts": "skill:typescript",
    "react": "skill:react",
    "reactjs": "skill:react",
    "react.js": "skill:react",
    "next": "skill:nextjs",
    "nextjs": "skill:nextjs",
    "next.js": "skill:nextjs",
    "vue": "skill:vue",
    "vuejs": "skill:vue",
    "angular": "skill:angular",
    "docker": "skill:docker",
    "kubernetes": "skill:kubernetes",
    "k8s": "skill:kubernetes",
    "aws": "skill:aws",
    "amazon web services": "skill:aws",
    "gcp": "skill:gcp",
    "google cloud": "skill:gcp",
    "azure": "skill:azure",
    "terraform": "skill:terraform",
    "ci/cd": "skill:cicd",
    "cicd": "skill:cicd",
    "sql": "skill:sql",
    "postgresql": "skill:postgresql",
    "postgres": "skill:postgresql",
    "mysql": "skill:mysql",
    "nosql": "skill:nosql",
    "mongodb": "skill:mongodb",
    "mongo": "skill:mongodb",
    "redis": "skill:redis",
    "spark": "skill:spark",
    "apache spark": "skill:spark",
    "kafka": "skill:kafka",
    "apache kafka": "skill:kafka",
    "airflow": "skill:airflow",
    "snowflake": "skill:snowflake",
}


def normalize_skill_name(name: str) -> str:
    """Normalize a skill name into a canonical ID."""
    raw = name.strip().lower()
    if raw.startswith("skill:"):
        raw = raw[6:]
    elif raw.startswith("domain:"):
        raw = raw[7:]

    clean = re.sub(r"[^\w\s\-\.]", "", raw).strip()
    if clean in SYNONYM_MAP:
        return SYNONYM_MAP[clean]

    # Remove dots and dashes
    compact = re.sub(r"[\s\-\.]+", "", clean)
    if compact in SYNONYM_MAP:
        return SYNONYM_MAP[compact]

    return f"skill:{compact}"


def skill_id_to_uri(skill_id: str) -> URIRef:
    """Convert skill ID to SKILL or CV namespace URI."""
    norm = normalize_skill_name(skill_id)
    clean = norm.replace("skill:", "").replace("domain:", "")
    return SKILL[clean]


def build_skos_ontology() -> Graph:
    """
    Build an RDFLib Graph defining recruitment ontologies,
    SKOS concepts, broader/narrower hierarchies, and classes.
    Standardizes prefixes cv:, skill:, skos:.
    """
    g = Graph()
    g.bind("cv", CV)
    g.bind("skill", SKILL)
    g.bind("skos", SKOS)
    g.bind("rdf", RDF)
    g.bind("rdfs", RDFS)
    g.bind("owl", OWL)

    # Core classes
    g.add((CV.Entity, RDF.type, RDFS.Class))
    g.add((CV.Candidate, RDFS.subClassOf, CV.Entity))
    g.add((CV.Role, RDFS.subClassOf, CV.Entity))
    g.add((CV.Skill, RDFS.subClassOf, CV.Entity))
    g.add((CV.Company, RDFS.subClassOf, CV.Entity))
    g.add((CV.Degree, RDFS.subClassOf, CV.Entity))
    g.add((CV.Project, RDFS.subClassOf, CV.Entity))

    # Core properties
    for pred in [
        CV.hasExperience,
        CV.usesSkill,
        CV.heldRole,
        CV.workedAt,
        CV.earnedDegree,
        CV.deliveredProject,
        CV.confidence,
        CV.sourceSnippet,
        CV.startDate,
        CV.endDate,
    ]:
        g.add((pred, RDF.type, RDF.Property))

    # Skill scheme
    scheme = CV.SkillTaxonomyScheme
    g.add((scheme, RDF.type, SKOS.ConceptScheme))
    g.add((scheme, RDFS.label, Literal("Recruitment Technical Skill Taxonomy", datatype=XSD.string)))

    # Populate SKOS hierarchy in SKILL and CV namespaces
    for child_id, broader_id, label in SKOS_SKILL_RELATIONS:
        child_clean = child_id.replace("skill:", "").replace("domain:", "")
        broader_clean = broader_id.replace("skill:", "").replace("domain:", "")

        child_skill_uri = SKILL[child_clean]
        broader_skill_uri = SKILL[broader_clean]

        child_cv_uri = CV[child_id.replace(":", "_")]
        broader_cv_uri = CV[broader_id.replace(":", "_")]

        for c_uri, b_uri in [(child_skill_uri, broader_skill_uri), (child_cv_uri, broader_cv_uri)]:
            g.add((c_uri, RDF.type, SKOS.Concept))
            g.add((c_uri, RDF.type, CV.Skill))
            g.add((c_uri, SKOS.inScheme, scheme))
            g.add((c_uri, SKOS.prefLabel, Literal(label, datatype=XSD.string)))
            g.add((c_uri, RDFS.label, Literal(label, datatype=XSD.string)))

            g.add((c_uri, SKOS.broader, b_uri))
            g.add((b_uri, SKOS.narrower, c_uri))
            g.add((c_uri, RDFS.subClassOf, b_uri))

    return g


# Cache singleton
_CACHED_TAXONOMY_NX: Optional[nx.Graph] = None


def get_taxonomy_networkx(g: Optional[Graph] = None) -> nx.Graph:
    """
    Export the SKOS skill hierarchy to an undirected NetworkX graph
    for computing shortest-path taxonomic distance.
    """
    global _CACHED_TAXONOMY_NX
    if g is None and _CACHED_TAXONOMY_NX is not None:
        return _CACHED_TAXONOMY_NX

    graph = nx.Graph()

    for child_id, broader_id, _ in SKOS_SKILL_RELATIONS:
        c_norm = normalize_skill_name(child_id)
        b_norm = normalize_skill_name(broader_id)
        graph.add_edge(c_norm, b_norm, weight=1.0)

    if g is not None:
        for s, p, o in g.triples((None, SKOS.broader, None)):
            s_name = str(s).split("#")[-1].replace("_", ":")
            o_name = str(o).split("#")[-1].replace("_", ":")
            graph.add_edge(normalize_skill_name(s_name), normalize_skill_name(o_name), weight=1.0)

    if g is None:
        _CACHED_TAXONOMY_NX = graph

    return graph


def calculate_taxonomic_distance(
    skill_a: str,
    skill_b: str,
    taxonomy_graph: Optional[nx.Graph] = None
) -> float:
    """
    Calculate shortest path distance between two skills in the SKOS taxonomy.
    """
    norm_a = normalize_skill_name(skill_a)
    norm_b = normalize_skill_name(skill_b)

    if norm_a == norm_b:
        return 0.0

    net = taxonomy_graph if taxonomy_graph is not None else get_taxonomy_networkx()

    if norm_a not in net or norm_b not in net:
        return float("inf")

    try:
        return float(nx.shortest_path_length(net, source=norm_a, target=norm_b))
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return float("inf")


def taxonomic_similarity(
    skill_a: str,
    skill_b: str,
    max_distance: int = 4,
    taxonomy_graph: Optional[nx.Graph] = None
) -> float:
    """
    Convert shortest-path distance to a similarity score [0.0, 1.0].
    """
    dist = calculate_taxonomic_distance(skill_a, skill_b, taxonomy_graph)
    if dist == 0.0:
        return 1.0
    if dist > max_distance or dist == float("inf"):
        return 0.0
    return float(1.0 / (1.0 + 0.4 * dist))
