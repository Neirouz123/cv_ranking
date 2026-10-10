"""
eval/generate_dataset.py
-------------------------
Generates the 24 synthetic CV profiles, the evaluation labels ground-truth matrix,
and the reference corpus for bilingual TF-IDF testing and ablation studies.
"""

import os
import csv
from pathlib import Path

CV_DIR = Path("eval/data/cvs")
CV_DIR.mkdir(parents=True, exist_ok=True)

CVS = {
    # -------------------------------------------------------------
    # JOB 1 TARGET CVs (Junior ML Engineer)
    # -------------------------------------------------------------
    "cv_01_ml_ideal.txt": """Lucas Bernard
Ingénieur Machine Learning Junior
lucas.bernard@example.com

Formation:
2020 - 2022: Master en Informatique et Intelligence Artificielle (Bac+5), Université Paris-Saclay.

Expérience Professionnelle:
2022-09 - 2024-06: Junior ML Engineer chez DeepTech Labs
- Développement et entraînement de modèles de Deep Learning en PyTorch et Python.
- Évaluation expérimentale avec Scikit-learn et versioning de code sous Git.
- Conteneurisation des modèles avec Docker pour les tests d'intégration.
""",

    "cv_02_ml_acceptable.txt": """Camille Robert
Data Scientist Junior
camille.robert@example.com

Formation:
2020 - 2023: Licence Informatique et Statistique (Bac+3), Université de Lyon.

Expérience Professionnelle:
2023-01 - 2024-02: Data Scientist chez Analytics France
- Entraînement de réseaux de neurones avec TensorFlow et Python.
- Modélisation statistique et classification avec Scikit-learn.
- Collaboration en équipe sous Git.
""",

    "cv_03_ml_marginal.txt": """Julien Mercier
Data Analyst
julien.mercier@example.com

Formation:
2021 - 2023: BTS Informatique (Bac+2).

Expérience Professionnelle:
2023-02 - 2024-03: Data Analyst chez Global Retail
- Création de tableaux de bord Excel et requêtes SQL complexes.
- Automatisation de scripts basiques en Python.
""",

    "cv_04_ml_offtopic.txt": """Guillaume Dufour
Chef de Cuisine
guillaume.dufour@example.com

Formation:
2015 - 2017: CAP Cuisine et Hôtellerie.

Expérience Professionnelle:
2018-01 - 2023-12: Chef de Partie puis Chef de Cuisine chez Le Bistrot Parisien
- Gestion complète de la brigade et élaboration des menus gastronomiques.
- Application rigoureuse des normes sanitaires HACCP et gestion des stocks.
""",

    "cv_05_ml_temporal_inversion.txt": """Alexandre Petit
Machine Learning Enthusiast
alexandre.petit@example.com

Formation:
2020 - 2022: Master Informatique (Bac+5).

Expérience Professionnelle:
2024-06 - 2021-01: ML Engineer chez Invertia Tech
- Développement de pipelines PyTorch et Python.
- Entraînement de modèles de Machine Learning.
""",

    "cv_06_ml_anachronism.txt": """Benoit Lemoine
Junior ML Developer
benoit.lemoine@example.com

Formation:
2021 - 2023: Master Data Science (Bac+5).

Expérience Professionnelle:
2011-01 - 2013-05: Machine Learning Intern chez RetroData
- Utilisation de FastAPI et PyTorch pour le déploiement d'API en production.
- Revendique 10 ans d'expérience intensive sur Python.
""",

    # -------------------------------------------------------------
    # JOB 2 TARGET CVs (Senior Data Platform Lead)
    # -------------------------------------------------------------
    "cv_07_data_ideal.txt": """Marc Lefebvre
Senior Data Platform Lead
marc.lefebvre@example.com

Formation:
2012 - 2017: Diplôme d'Ingénieur en Génie Informatique (Bac+5), INSA Lyon.

Expérience Professionnelle:
2017-09 - 2020-12: Data Engineer Senior chez BigData Corp
- Pipeline de traitement distribué massif avec Apache Spark (PySpark) et Python.
- Déploiement et orchestration de microservices sur clusters Kubernetes.
2021-01 - 2024-06: Lead Data Platform chez CloudScale Solutions
- Architecture de données streaming avec Apache Kafka et AWS.
- Industrialisation CI/CD et infrastructure as code avec Docker et Kubernetes.
""",

    "cv_08_data_acceptable.txt": """Sarah Benali
Data Engineer Confirmée
sarah.benali@example.com

Formation:
2015 - 2020: Master Informatique Décisionnelle (Bac+5), Université Paris Cité.

Expérience Professionnelle:
2020-03 - 2024-05: Data Engineer chez FinData Services
- Traitement batch à grande échelle avec Apache Spark et Python.
- Ingestion de flux de données via Apache Kafka et conteneurisation Docker sur GCP.
""",

    "cv_09_data_marginal.txt": """Hugo Martin
Développeur Backend Junior
hugo.martin@example.com

Formation:
2021 - 2023: Licence Informatique (Bac+3).

Expérience Professionnelle:
2023-01 - 2024-06: Développeur Backend chez WebServices
- Développement d'APIs REST en Python avec PostgreSQL.
- Maintenance de scripts d'extraction de données.
""",

    "cv_10_data_offtopic.txt": """Claire Dubois
Responsable Comptable et Trésorerie
claire.dubois@example.com

Formation:
2013 - 2018: Master Comptabilité, Contrôle, Audit (Bac+5).

Expérience Professionnelle:
2018-09 - 2024-05: Responsable Comptable chez Audit Conseil
- Établissement des liasses fiscales, clôture mensuelle et bilan annuel sous Sage.
- Gestion de la trésorerie et reporting financier avec Excel.
""",

    "cv_11_data_missing_must.txt": """David Moreau
Senior Cloud Architect
david.moreau@example.com

Formation:
2013 - 2018: Diplôme d'Ingénieur Réseaux et Cloud (Bac+5).

Expérience Professionnelle:
2018-06 - 2024-06: Cloud Infrastructure Architect chez CloudNative Inc
- Conception d'architectures cloud haute disponibilité sur AWS.
- Automatisation et streaming avec Apache Kafka et scripts Python.
- Absence d'expérience sur Apache Spark et orchestration Kubernetes.
""",

    "cv_12_data_temporal_conflict.txt": """Thomas Rousseau
Data Platform Engineer
thomas.rousseau@example.com

Formation:
2014 - 2019: Master Big Data (Bac+5).

Expérience Professionnelle:
2025-01 - 2018-06: Lead Big Data chez Paradoxal Systems
- Traitement distribué sous Apache Spark, Kubernetes et Python.
- Pipeline Kafka en production.
""",

    # -------------------------------------------------------------
    # JOB 3 TARGET CVs (Embedded Systems & IoT Developer)
    # -------------------------------------------------------------
    "cv_13_iot_ideal.txt": """Romain Garnier
Ingénieur Systèmes Embarqués & IoT
romain.garnier@example.com

Formation:
2015 - 2020: Diplôme d'Ingénieur en Systèmes Embarqués (Bac+5), ENSEA.

Expérience Professionnelle:
2020-09 - 2024-06: Ingénieur Embarqué chez IoT Innovations
- Conception firmware en langage C et C++ sur cibles microcontrôleurs ARM Cortex.
- Programmation multitâche temps réel avec l'OS FreeRTOS (tâches, mutex, queues).
- Implémentation du protocole MQTT pour capteurs industriels connectés.
- Environnement de build et tests sous Linux embarqué.
""",

    "cv_14_iot_acceptable.txt": """Elise Fournier
Développeuse Microcontrôleurs
elise.fournier@example.com

Formation:
2018 - 2021: Licence Professionnelle Systèmes Électroniques (Bac+3).

Expérience Professionnelle:
2021-09 - 2024-03: Développeuse Firmware chez ElectroSens
- Programmation en langage C pour cibles STM32 et microcontrôleurs.
- Gestion de bus de communication UART, SPI, I2C sous environnement Linux.
""",

    "cv_15_iot_marginal.txt": """Valentin Morel
Développeur Frontend
valentin.morel@example.com

Formation:
2020 - 2023: Bachelor Métiers du Web (Bac+3).

Expérience Professionnelle:
2023-01 - 2024-05: Intégrateur Web chez Studio Créatif
- Développement d'interfaces utilisateur avec React et CSS.
- Expérimentation personnelle sur cartes Arduino de loisir.
""",

    "cv_16_iot_offtopic.txt": """Emilie Roy
Juriste Droit des Affaires
emilie.roy@example.com

Formation:
2015 - 2020: Master en Droit Privé et Droit des Affaires (Bac+5).

Expérience Professionnelle:
2020-10 - 2024-04: Juriste Entreprise chez Global Legal
- Rédaction et négociation de contrats commerciaux et conformité RGPD.
- Gestion des contentieux et veille juridique réglementaire.
""",

    "cv_17_iot_missing_must.txt": """Mathieu Blanc
Concepteur Hardware & PCB
mathieu.blanc@example.com

Formation:
2016 - 2020: Master Électronique de Puissance (Bac+5).

Expérience Professionnelle:
2020-07 - 2024-06: Ingénieur CAO Électronique chez BoardDesign
- Conception de circuits imprimés multicouches sous Altium Designer.
- Routage haute fréquence et bancs de test matériels.
- Aucune compétence en programmation firmware C ou FreeRTOS.
""",

    "cv_18_iot_anachronism.txt": """Nicolas Chevalier
Firmware Engineer
nicolas.chevalier@example.com

Formation:
2015 - 2020: Ingénieur Électronique (Bac+5).

Expérience Professionnelle:
2007-01 - 2009-12: Embedded Engineer chez RetroEmbedded
- Développement firmware C sous FreeRTOS avec conteneurisation Docker.
""",

    # -------------------------------------------------------------
    # JOB 4 TARGET CVs (Fullstack Web Developer React / FastAPI)
    # -------------------------------------------------------------
    "cv_19_fullstack_ideal.txt": """Maxime Gauthier
Senior Fullstack Developer
maxime.gauthier@example.com

Formation:
2015 - 2020: Master en Ingénierie Logicielle (Bac+5), Université de Rennes.

Expérience Professionnelle:
2020-09 - 2024-06: Fullstack Developer chez SaaS Platform (Bilingual Team FR/EN)
- Frontend architecture using React and TypeScript for dynamic client interfaces.
- Backend API REST development with FastAPI and Python.
- Database modeling and querying with PostgreSQL.
- Docker containers for local development and CI/CD deployment pipelines.
""",

    "cv_20_fullstack_acceptable.txt": """Sophie Delattre
Développeuse Web Fullstack
sophie.delattre@example.com

Formation:
2018 - 2021: Licence Informatique Web & Mobile (Bac+3).

Expérience Professionnelle:
2021-09 - 2024-05: Développeuse Fullstack chez Digital Agency
- Développement d'applications web interactives avec React.
- Création de serveurs backend avec Node.js, Express et bases PostgreSQL.
- Utilisation quotidienne de Docker pour les environnements de test.
""",

    "cv_21_fullstack_marginal.txt": """Florian Brunet
Intégrateur Web PHP / WordPress
florian.brunet@example.com

Formation:
2019 - 2021: BTS Services Informatiques aux Organisations (Bac+2).

Expérience Professionnelle:
2021-07 - 2024-05: Intégrateur CMS chez Agence Com
- Création de sites vitrines sous WordPress et modules PHP.
- Stylisation HTML5, CSS3 et interactions jQuery légères.
- Bases de données MySQL basiques.
""",

    "cv_22_fullstack_offtopic.txt": """Antoine Caron
Infirmier Diplômé d'État
antoine.caron@example.com

Formation:
2016 - 2019: Diplôme d'État d'Infirmier (Bac+3), IFSI Paris.

Expérience Professionnelle:
2019-10 - 2024-05: Infirmier en Soins Intensifs au CHU
- Prise en charge globale des patients en service de réanimation.
- Administration des thérapeutiques médicamenteuses et surveillance vitale.
""",

    "cv_23_fullstack_missing_must.txt": """Sebastien Renaud
Lead Backend Python
sebastien.renaud@example.com

Formation:
2015 - 2020: Master Informatique (Bac+5).

Expérience Professionnelle:
2020-08 - 2024-06: Backend Architect chez DataWeb
- Développement d'infrastructures d'APIs performantes avec FastAPI et Python.
- Optimisation de requêtes SQL sur PostgreSQL et clusters Docker.
- Aucune compétence frontend en React ou TypeScript.
""",

    "cv_24_fullstack_perturbed.txt": """Damien Colin
Fullstack Developer
damien.colin@example.com

Formation:
2016 - 2021: Master Génie Logiciel (Bac+5).

Expérience Professionnelle:
2024-06 - 2019-01: Fullstack Lead chez InvertWeb
- Utilisation intensive de React et FastAPI pour des applications distribuées.
- Utilisation de FastAPI déclarée en 2014 sur des projets d'études.
""",
}

for filename, content in CVS.items():
    path = CV_DIR / filename
    path.write_text(content.strip() + "\n", encoding="utf-8")
    print(f"Created {path}")

# -------------------------------------------------------------
# LABELS GROUND TRUTH MATRIX (4 jobs x 24 CVs = 96 ratings)
# -------------------------------------------------------------
# Ratings:
# 3: Ideal match (Tier 3)
# 2: Acceptable match (Tier 2)
# 1: Marginal or Perturbed match (Tier 1)
# 0: Off-topic / Irrelevant (Tier 0)

JOBS = [
    "job_1_junior_ml",
    "job_2_senior_data_lead",
    "job_3_embedded_iot",
    "job_4_fullstack_fr_en",
]

# Baseline ratings mapping
# (job_id, cv_id) -> relevance_grade
RATINGS = {
    # Job 1 (Junior ML)
    ("job_1_junior_ml", "cv_01_ml_ideal.txt"): 3,
    ("job_1_junior_ml", "cv_02_ml_acceptable.txt"): 2,
    ("job_1_junior_ml", "cv_03_ml_marginal.txt"): 1,
    ("job_1_junior_ml", "cv_04_ml_offtopic.txt"): 0,
    ("job_1_junior_ml", "cv_05_ml_temporal_inversion.txt"): 1,
    ("job_1_junior_ml", "cv_06_ml_anachronism.txt"): 1,
    ("job_1_junior_ml", "cv_07_data_ideal.txt"): 1, # Senior Data Lead has Python but overqualified/different domain
    ("job_1_junior_ml", "cv_08_data_acceptable.txt"): 1,
    ("job_1_junior_ml", "cv_09_data_marginal.txt"): 0,
    ("job_1_junior_ml", "cv_10_data_offtopic.txt"): 0,
    ("job_1_junior_ml", "cv_11_data_missing_must.txt"): 0,
    ("job_1_junior_ml", "cv_12_data_temporal_conflict.txt"): 0,
    ("job_1_junior_ml", "cv_13_iot_ideal.txt"): 0,
    ("job_1_junior_ml", "cv_14_iot_acceptable.txt"): 0,
    ("job_1_junior_ml", "cv_15_iot_marginal.txt"): 0,
    ("job_1_junior_ml", "cv_16_iot_offtopic.txt"): 0,
    ("job_1_junior_ml", "cv_17_iot_missing_must.txt"): 0,
    ("job_1_junior_ml", "cv_18_iot_anachronism.txt"): 0,
    ("job_1_junior_ml", "cv_19_fullstack_ideal.txt"): 0,
    ("job_1_junior_ml", "cv_20_fullstack_acceptable.txt"): 0,
    ("job_1_junior_ml", "cv_21_fullstack_marginal.txt"): 0,
    ("job_1_junior_ml", "cv_22_fullstack_offtopic.txt"): 0,
    ("job_1_junior_ml", "cv_23_fullstack_missing_must.txt"): 0,
    ("job_1_junior_ml", "cv_24_fullstack_perturbed.txt"): 0,

    # Job 2 (Senior Data Platform Lead)
    ("job_2_senior_data_lead", "cv_01_ml_ideal.txt"): 0, # Junior ML lacks Spark, K8s, seniority
    ("job_2_senior_data_lead", "cv_02_ml_acceptable.txt"): 0,
    ("job_2_senior_data_lead", "cv_03_ml_marginal.txt"): 0,
    ("job_2_senior_data_lead", "cv_04_ml_offtopic.txt"): 0,
    ("job_2_senior_data_lead", "cv_05_ml_temporal_inversion.txt"): 0,
    ("job_2_senior_data_lead", "cv_06_ml_anachronism.txt"): 0,
    ("job_2_senior_data_lead", "cv_07_data_ideal.txt"): 3,
    ("job_2_senior_data_lead", "cv_08_data_acceptable.txt"): 2,
    ("job_2_senior_data_lead", "cv_09_data_marginal.txt"): 1,
    ("job_2_senior_data_lead", "cv_10_data_offtopic.txt"): 0,
    ("job_2_senior_data_lead", "cv_11_data_missing_must.txt"): 1,
    ("job_2_senior_data_lead", "cv_12_data_temporal_conflict.txt"): 1,
    ("job_2_senior_data_lead", "cv_13_iot_ideal.txt"): 0,
    ("job_2_senior_data_lead", "cv_14_iot_acceptable.txt"): 0,
    ("job_2_senior_data_lead", "cv_15_iot_marginal.txt"): 0,
    ("job_2_senior_data_lead", "cv_16_iot_offtopic.txt"): 0,
    ("job_2_senior_data_lead", "cv_17_iot_missing_must.txt"): 0,
    ("job_2_senior_data_lead", "cv_18_iot_anachronism.txt"): 0,
    ("job_2_senior_data_lead", "cv_19_fullstack_ideal.txt"): 0,
    ("job_2_senior_data_lead", "cv_20_fullstack_acceptable.txt"): 0,
    ("job_2_senior_data_lead", "cv_21_fullstack_marginal.txt"): 0,
    ("job_2_senior_data_lead", "cv_22_fullstack_offtopic.txt"): 0,
    ("job_2_senior_data_lead", "cv_23_fullstack_missing_must.txt"): 1,
    ("job_2_senior_data_lead", "cv_24_fullstack_perturbed.txt"): 0,

    # Job 3 (Embedded Systems & IoT)
    ("job_3_embedded_iot", "cv_01_ml_ideal.txt"): 0,
    ("job_3_embedded_iot", "cv_02_ml_acceptable.txt"): 0,
    ("job_3_embedded_iot", "cv_03_ml_marginal.txt"): 0,
    ("job_3_embedded_iot", "cv_04_ml_offtopic.txt"): 0,
    ("job_3_embedded_iot", "cv_05_ml_temporal_inversion.txt"): 0,
    ("job_3_embedded_iot", "cv_06_ml_anachronism.txt"): 0,
    ("job_3_embedded_iot", "cv_07_data_ideal.txt"): 0,
    ("job_3_embedded_iot", "cv_08_data_acceptable.txt"): 0,
    ("job_3_embedded_iot", "cv_09_data_marginal.txt"): 0,
    ("job_3_embedded_iot", "cv_10_data_offtopic.txt"): 0,
    ("job_3_embedded_iot", "cv_11_data_missing_must.txt"): 0,
    ("job_3_embedded_iot", "cv_12_data_temporal_conflict.txt"): 0,
    ("job_3_embedded_iot", "cv_13_iot_ideal.txt"): 3,
    ("job_3_embedded_iot", "cv_14_iot_acceptable.txt"): 2,
    ("job_3_embedded_iot", "cv_15_iot_marginal.txt"): 1,
    ("job_3_embedded_iot", "cv_16_iot_offtopic.txt"): 0,
    ("job_3_embedded_iot", "cv_17_iot_missing_must.txt"): 1,
    ("job_3_embedded_iot", "cv_18_iot_anachronism.txt"): 1,
    ("job_3_embedded_iot", "cv_19_fullstack_ideal.txt"): 0,
    ("job_3_embedded_iot", "cv_20_fullstack_acceptable.txt"): 0,
    ("job_3_embedded_iot", "cv_21_fullstack_marginal.txt"): 0,
    ("job_3_embedded_iot", "cv_22_fullstack_offtopic.txt"): 0,
    ("job_3_embedded_iot", "cv_23_fullstack_missing_must.txt"): 0,
    ("job_3_embedded_iot", "cv_24_fullstack_perturbed.txt"): 0,

    # Job 4 (Fullstack React / FastAPI)
    ("job_4_fullstack_fr_en", "cv_01_ml_ideal.txt"): 0,
    ("job_4_fullstack_fr_en", "cv_02_ml_acceptable.txt"): 0,
    ("job_4_fullstack_fr_en", "cv_03_ml_marginal.txt"): 0,
    ("job_4_fullstack_fr_en", "cv_04_ml_offtopic.txt"): 0,
    ("job_4_fullstack_fr_en", "cv_05_ml_temporal_inversion.txt"): 0,
    ("job_4_fullstack_fr_en", "cv_06_ml_anachronism.txt"): 0,
    ("job_4_fullstack_fr_en", "cv_07_data_ideal.txt"): 0,
    ("job_4_fullstack_fr_en", "cv_08_data_acceptable.txt"): 0,
    ("job_4_fullstack_fr_en", "cv_09_data_marginal.txt"): 1, # Junior backend Python/Postgres
    ("job_4_fullstack_fr_en", "cv_10_data_offtopic.txt"): 0,
    ("job_4_fullstack_fr_en", "cv_11_data_missing_must.txt"): 0,
    ("job_4_fullstack_fr_en", "cv_12_data_temporal_conflict.txt"): 0,
    ("job_4_fullstack_fr_en", "cv_13_iot_ideal.txt"): 0,
    ("job_4_fullstack_fr_en", "cv_14_iot_acceptable.txt"): 0,
    ("job_4_fullstack_fr_en", "cv_15_iot_marginal.txt"): 1,
    ("job_4_fullstack_fr_en", "cv_16_iot_offtopic.txt"): 0,
    ("job_4_fullstack_fr_en", "cv_17_iot_missing_must.txt"): 0,
    ("job_4_fullstack_fr_en", "cv_18_iot_anachronism.txt"): 0,
    ("job_4_fullstack_fr_en", "cv_19_fullstack_ideal.txt"): 3,
    ("job_4_fullstack_fr_en", "cv_20_fullstack_acceptable.txt"): 2,
    ("job_4_fullstack_fr_en", "cv_21_fullstack_marginal.txt"): 1,
    ("job_4_fullstack_fr_en", "cv_22_fullstack_offtopic.txt"): 0,
    ("job_4_fullstack_fr_en", "cv_23_fullstack_missing_must.txt"): 1,
    ("job_4_fullstack_fr_en", "cv_24_fullstack_perturbed.txt"): 1,
}

labels_path = Path("eval/labels.csv")
with open(labels_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["job_id", "cv_id", "relevance_grade"])
    for (jid, cid), grade in sorted(RATINGS.items()):
        writer.writerow([jid, cid, grade])

print(f"Created {labels_path} with {len(RATINGS)} rows.")

# -------------------------------------------------------------
# REFERENCE CORPUS (bilingual IT/Tech passages for IDF stats)
# -------------------------------------------------------------
corpus_path = Path("eval/data/reference_corpus.txt")
reference_texts = [
    "Ingénieur logiciel senior spécialisé en architecture microservices et conteneurs Docker et Kubernetes.",
    "Data Scientist avec expertise en apprentissage automatique, réseaux de neurones PyTorch et traitement du langage naturel.",
    "Développeur Fullstack expérimenté en TypeScript, React, Next.js, Node.js et bases de données PostgreSQL.",
    "Lead Cloud DevOps maîtrisant Terraform, AWS, Azure, pipelines CI/CD et surveillance Prometheus Grafana.",
    "Développeur systèmes embarqués bas niveau en langage C et C++, RTOS, FreeRTOS, microcontrôleurs STM32 et bus CAN.",
    "Chef de projet technique agile Scrum Master assurant la coordination des équipes de développement logiciel et les livrables.",
    "Data Engineer expérimenté en streaming de données temps réel avec Apache Kafka, Apache Spark, Databricks et Snowflake.",
    "Expert en cybersécurité, analyse des vulnérabilités, tests d'intrusion, protocoles TLS et conformité ISO 27001.",
    "Administrateur systèmes et réseaux Linux RedHat, Debian, virtualisation VMware et automatisation Ansible.",
    "Développeur backend Python FastAPI et Django, conception d'APIs REST sécurisées et intégration continue.",
]
corpus_path.write_text("\n\n".join(reference_texts), encoding="utf-8")
print(f"Created {corpus_path}")
