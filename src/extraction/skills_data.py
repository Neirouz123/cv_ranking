"""
skills_data.py
---------------
Curated skills taxonomy used to detect skills in résumés and job offers.

Structure: { category: { canonical_skill_name: [synonyms / patterns, ...] } }

- Canonical names are what gets displayed in the UI (matched/missing skills).
- Synonyms should include common French and English variants, abbreviations,
  and frequent misspellings/spacing variants. Matching is case-insensitive
  and accent-insensitive (see extractor.py), and uses word boundaries, so
  short synonyms like "r" or "go" are intentionally written with enough
  context (handled via regex word boundaries) to avoid false positives.

This taxonomy is intentionally editable: HR teams can add role-specific
skills (e.g. industry certifications, internal tools) without touching any
other part of the codebase.
"""

SKILLS_TAXONOMY: dict[str, dict[str, list[str]]] = {
    "Langages de programmation": {
        "Python": ["python"],
        "JavaScript": ["javascript", "js", "es6", "ecmascript"],
        "TypeScript": ["typescript", "ts"],
        "Java": ["java(?!script)"],
        "C#": ["c#", "csharp", "\\.net"],
        "C++": ["c\\+\\+", "cpp"],
        "C": ["\\bc\\b(?!\\+\\+|#)"],
        "PHP": ["php"],
        "Ruby": ["ruby"],
        "Go": ["golang", "\\bgo\\b"],
        "Rust": ["rust"],
        "Swift": ["swift"],
        "Kotlin": ["kotlin"],
        "SQL": ["sql"],
        "R": ["\\br\\b(?=.*(statistiqu|data|analys|studio))", "r studio", "rstudio"],
        "Scala": ["scala"],
        "Bash / Shell": ["bash", "shell script", "shell scripting"],
        "MATLAB": ["matlab"],
    },
    "Frameworks & bibliothèques": {
        "React": ["react\\.?js", "react native", "\\breact\\b"],
        "Angular": ["angular"],
        "Vue.js": ["vue\\.?js", "\\bvue\\b"],
        "Node.js": ["node\\.?js", "nodejs"],
        "Express.js": ["express\\.?js"],
        "Django": ["django"],
        "Flask": ["flask"],
        "FastAPI": ["fastapi"],
        "Spring / Spring Boot": ["spring boot", "spring framework", "\\bspring\\b"],
        ".NET": ["\\.net", "asp\\.net", "dotnet"],
        "Laravel": ["laravel"],
        "Symfony": ["symfony"],
        "Ruby on Rails": ["ruby on rails", "\\brails\\b"],
        "jQuery": ["jquery"],
        "Bootstrap": ["bootstrap"],
        "Tailwind CSS": ["tailwind"],
        "Next.js": ["next\\.?js"],
        "Pandas": ["pandas"],
        "NumPy": ["numpy"],
        "TensorFlow": ["tensorflow"],
        "PyTorch": ["pytorch"],
        "Scikit-learn": ["scikit-learn", "sklearn"],
    },
    "Bases de données": {
        "MySQL": ["mysql"],
        "PostgreSQL": ["postgresql", "postgres"],
        "MongoDB": ["mongodb", "mongo"],
        "Oracle": ["oracle db", "oracle database", "\\boracle\\b"],
        "SQL Server": ["sql server", "mssql"],
        "SQLite": ["sqlite"],
        "Redis": ["redis"],
        "Elasticsearch": ["elasticsearch", "elastic search"],
        "Cassandra": ["cassandra"],
        "MariaDB": ["mariadb"],
    },
    "Cloud & DevOps": {
        "AWS": ["amazon web services", "\\baws\\b"],
        "Microsoft Azure": ["\\bazure\\b"],
        "Google Cloud (GCP)": ["google cloud", "\\bgcp\\b"],
        "Docker": ["docker"],
        "Kubernetes": ["kubernetes", "\\bk8s\\b"],
        "Terraform": ["terraform"],
        "Ansible": ["ansible"],
        "Jenkins": ["jenkins"],
        "GitLab CI/CD": ["gitlab ci", "gitlab-ci"],
        "CI/CD": ["ci/cd", "continuous integration", "intégration continue"],
        "Git": ["\\bgit\\b"],
        "Linux": ["linux", "unix"],
        "Nginx": ["nginx"],
    },
    "Data & IA": {
        "Machine Learning": ["machine learning", "apprentissage automatique", "\\bml\\b"],
        "Deep Learning": ["deep learning", "apprentissage profond"],
        "Data Science": ["data science", "science des données"],
        "Data Analysis": ["data analysis", "analyse de données"],
        "Big Data": ["big data"],
        "NLP": ["nlp", "traitement du langage naturel", "natural language processing"],
        "Power BI": ["power bi", "powerbi"],
        "Tableau": ["tableau software", "\\btableau\\b(?=.*(data|dashboard|bi|visualis))"],
        "Excel avancé": ["excel avancé", "advanced excel", "tcd", "tableau croisé dynamique"],
        "ETL": ["\\betl\\b", "extract transform load"],
        "Spark": ["apache spark", "\\bspark\\b"],
        "Hadoop": ["hadoop"],
    },
    "Gestion de projet & méthodologie": {
        "Agile": ["agile"],
        "Scrum": ["scrum", "scrum master"],
        "Kanban": ["kanban"],
        "SAFe": ["\\bsafe\\b(?=.*(agile|framework))"],
        "PMP": ["\\bpmp\\b", "project management professional"],
        "Prince2": ["prince2", "prince 2"],
        "Jira": ["jira"],
        "Trello": ["trello"],
        "MS Project": ["ms project", "microsoft project"],
        "Gestion de projet": ["gestion de projet", "project management"],
        "Cahier des charges": ["cahier des charges", "spécifications fonctionnelles"],
    },
    "Design & Produit": {
        "Figma": ["figma"],
        "Adobe XD": ["adobe xd"],
        "Sketch": ["sketch"],
        "Photoshop": ["photoshop"],
        "Illustrator": ["illustrator"],
        "UX/UI Design": ["ux/ui", "user experience", "user interface", "\\bux\\b", "\\bui\\b"],
        "Product Management": ["product management", "product owner", "gestion de produit"],
        "Design Thinking": ["design thinking"],
        "Prototypage": ["prototypage", "prototyping", "wireframe", "maquettage"],
    },
    "Marketing & Communication": {
        "SEO": ["\\bseo\\b", "référencement naturel"],
        "SEA": ["\\bsea\\b", "référencement payant", "google ads"],
        "Réseaux sociaux": ["réseaux sociaux", "social media", "community management"],
        "Google Analytics": ["google analytics"],
        "Content Marketing": ["content marketing", "marketing de contenu"],
        "Email Marketing": ["email marketing", "emailing"],
        "CRM": ["\\bcrm\\b", "salesforce", "hubspot"],
        "Marketing Automation": ["marketing automation"],
        "Copywriting": ["copywriting", "rédaction web"],
    },
    "Langues (compétences linguistiques)": {
        "Français": ["français courant", "français natif", "native french", "fluent french"],
        "Anglais": ["anglais courant", "anglais professionnel", "fluent english",
                    "professional english", "toeic", "toefl", "ielts"],
        "Arabe": ["arabe courant", "arabe natif", "fluent arabic", "native arabic"],
        "Espagnol": ["espagnol courant", "fluent spanish"],
        "Allemand": ["allemand courant", "fluent german"],
        "Italien": ["italien courant", "fluent italian"],
        "Mandarin": ["mandarin", "chinois courant"],
    },
    "Compétences transversales (soft skills)": {
        "Travail en équipe": ["travail en équipe", "esprit d'équipe", "teamwork", "team player"],
        "Communication": ["communication", "sens de la communication"],
        "Leadership": ["leadership", "prise de décision", "encadrement d'équipe",
                        "management d'équipe", "team management"],
        "Autonomie": ["autonomie", "autonome", "self-starter"],
        "Résolution de problèmes": ["résolution de problèmes", "problem solving",
                                     "esprit d'analyse", "analytical thinking"],
        "Rigueur": ["rigueur", "rigoureux", "attention to detail"],
        "Adaptabilité": ["adaptabilité", "adaptability", "flexibilité"],
        "Gestion du temps": ["gestion du temps", "time management", "sens des priorités"],
        "Créativité": ["créativité", "creativity", "force de proposition"],
        "Négociation": ["négociation", "negotiation"],
    },
    "Certifications": {
        "AWS Certified": ["aws certified", "aws certification"],
        "Certification Scrum": ["certified scrummaster", "\\bcsm\\b", "psm i", "psm 1"],
        "ITIL": ["\\bitil\\b"],
        "CISSP": ["cissp"],
        "CFA": ["\\bcfa\\b(?=.*(finance|financial))"],
        "Google Certified": ["google certified"],
        "Microsoft Certified": ["microsoft certified"],
        "PMI": ["\\bpmi\\b"],
    },
}


def all_skills_flat() -> dict[str, list[str]]:
    """Return a flat {canonical_skill: [patterns]} dict across all categories."""
    flat: dict[str, list[str]] = {}
    for category_skills in SKILLS_TAXONOMY.values():
        flat.update(category_skills)
    return flat


def skill_category(skill_name: str) -> str | None:
    """Look up which category a canonical skill name belongs to."""
    for category, skills in SKILLS_TAXONOMY.items():
        if skill_name in skills:
            return category
    return None
