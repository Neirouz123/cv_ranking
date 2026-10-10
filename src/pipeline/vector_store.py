"""
src/pipeline/vector_store.py
----------------------------
Robust bilingual (French / English) TF-IDF corpus vectorizer.
Provides:
1. Diacritic stripping, email/URL removal, and text normalization.
2. Merged bilingual French and English stopword dictionaries.
3. Multi-document corpus-level fitting with optional offline reference corpus.
4. Pure, IDF-penalized cosine similarity without heuristic noise overrides.
"""

from __future__ import annotations

import os
import re
import unicodedata
from pathlib import Path
from typing import Optional, Sequence

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Comprehensive French stopword lexicon
FRENCH_STOPWORDS = frozenset({
    "a", "à", "au", "aux", "avec", "ce", "ces", "cet", "cette", "dans", "de",
    "des", "du", "elle", "en", "et", "eux", "il", "ils", "je", "la", "le",
    "les", "leur", "leurs", "lui", "ma", "mais", "me", "même", "mes", "moi",
    "mon", "ne", "nos", "notre", "nous", "on", "ou", "où", "par", "pas",
    "pour", "qu", "que", "qui", "sa", "se", "ses", "son", "sur", "ta", "te",
    "tes", "toi", "ton", "tous", "tout", "toute", "toutes", "tu", "un", "une",
    "vos", "votre", "vous", "c", "d", "j", "l", "m", "n", "s", "t", "y",
    "été", "étée", "étées", "étés", "étant", "suis", "es", "est", "sommes",
    "êtes", "sont", "serai", "seras", "sera", "serons", "serez", "seront",
    "serais", "serait", "serions", "seriez", "seraient", "étais", "était",
    "étions", "étiez", "étaient", "fus", "fut", "fûmes", "fûtes", "furent",
    "sois", "soit", "soyons", "soyez", "soient", "fusse", "fusses", "fût",
    "fussions", "fussiez", "fussent", "ayant", "ayante", "ayantes", "ayants",
    "ai", "as", "avons", "avez", "ont", "aurai", "auras", "aura", "aurons",
    "aurez", "auront", "aurais", "aurait", "aurions", "auriez", "auraient",
    "avais", "avait", "avions", "aviez", "avaient", "eut", "eûmes", "eûtes",
    "eurent", "aie", "aies", "ait", "ayons", "ayez", "aient", "eusse",
    "eusses", "eût", "eussions", "eussiez", "eussent", "afin", "alors",
    "après", "ainsi", "aussi", "autre", "autres", "avant", "bien", "car",
    "chez", "comme", "donc", "dont", "dès", "entre", "faire", "fait", "jusque",
    "moins", "non", "notamment", "parce", "pendant", "plus", "plusieurs",
    "pouvoir", "puis", "quand", "sans", "selon", "si", "sous", "tant", "très",
    "vers", "via",
})

# Merged bilingual stopword set
BILINGUAL_STOPWORDS = frozenset(ENGLISH_STOP_WORDS.union(FRENCH_STOPWORDS))


def sanitize_text(text: Optional[str]) -> str:
    """
    Sanitize text for clean lexical matching:
    1. Lowercase text.
    2. Strip diacritics/accents (e.g. 'ingénieur' -> 'ingenieur').
    3. Strip URLs, emails, and isolated numbers.
    4. Remove non-alphanumeric punctuation while preserving technical tokens (c++, c#, .net).
    """
    if not text:
        return ""

    raw = str(text)

    # 1. Strip URLs and email addresses
    raw = re.sub(r"https?://\S+|www\.\S+", " ", raw)
    raw = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", " ", raw)

    # 2. Normalize and strip diacritics
    nfkd = unicodedata.normalize("NFKD", raw)
    stripped = "".join(c for c in nfkd if not unicodedata.combining(c))

    # 3. Lowercase
    cleaned = stripped.lower()

    # 4. Protect specific tech keywords before punctuation stripping
    cleaned = cleaned.replace("c++", "cpp").replace("c#", "csharp").replace(".net", "dotnet")

    # 5. Remove isolated numbers and punctuation except words
    cleaned = re.sub(r"\b\d+\b", " ", cleaned)
    cleaned = re.sub(r"[^\w\s]", " ", cleaned)

    # 6. Normalize whitespace
    words = [w for w in cleaned.split() if len(w) > 1]
    return " ".join(words)


class CorpusVectorizer:
    """
    Bilingual Corpus-level TF-IDF Vectorizer.
    Fitted across candidate documents + job descriptions, supplemented by an
    offline reference corpus to guarantee statistical IDF stability.
    """

    def __init__(
        self,
        reference_corpus_path: Optional[str] = "eval/data/reference_corpus.txt",
        ngram_range: tuple[int, int] = (1, 2),
        min_df: int = 1,
    ) -> None:
        self.reference_corpus_path = reference_corpus_path
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.vectorizer: Optional[TfidfVectorizer] = None
        self._reference_docs: list[str] = self._load_reference_corpus()

    def _load_reference_corpus(self) -> list[str]:
        """Loads offline reference documents for robust IDF computation."""
        docs: list[str] = []
        if self.reference_corpus_path and Path(self.reference_corpus_path).exists():
            try:
                content = Path(self.reference_corpus_path).read_text(encoding="utf-8")
                paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
                docs.extend([sanitize_text(p) for p in paragraphs if p])
            except Exception:
                pass

        if not docs:
            # Fallback built-in reference documents if file is missing
            docs = [
                sanitize_text("Ingenieur logiciel senior developpement python machine learning kubernetes"),
                sanitize_text("Data scientist intelligence artificielle modeles profonds pytorch scikit learn"),
                sanitize_text("Developpeur fullstack web react typescript fastapi postgresql docker api rest"),
                sanitize_text("Systemes embarques programmation c microcontroleurs rtos freertos mqtt arm"),
                sanitize_text("Architecte cloud aws devops streaming kafka apache spark pipelines big data"),
            ]
        return docs

    def fit(self, documents: Sequence[str]) -> "CorpusVectorizer":
        """
        Fits TF-IDF across the given document collection merged with reference corpus.
        """
        sanitized_docs = [sanitize_text(d) for d in documents if d and d.strip()]
        # Merge with reference docs to guarantee sufficient document frequency support
        full_corpus = list(sanitized_docs) + self._reference_docs

        self.vectorizer = TfidfVectorizer(
            stop_words=list(BILINGUAL_STOPWORDS),
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            norm="l2",
        )
        self.vectorizer.fit(full_corpus)
        return self

    def compute_similarity(
        self,
        text_a: Optional[str],
        text_b: Optional[str],
        additional_context: Optional[Sequence[str]] = None,
    ) -> float:
        """
        Computes pure cosine similarity between text_a and text_b.
        If the vectorizer is not already fitted, fits across [text_a, text_b] + context.
        """
        if not text_a or not text_b:
            return 0.0

        clean_a = sanitize_text(text_a)
        clean_b = sanitize_text(text_b)

        if not clean_a or not clean_b:
            return 0.0

        if self.vectorizer is None:
            docs_to_fit = [clean_a, clean_b]
            if additional_context:
                docs_to_fit.extend([sanitize_text(c) for c in additional_context if c])
            self.fit(docs_to_fit)

        try:
            tfidf_mat = self.vectorizer.transform([clean_a, clean_b])
            cos_sim = float(cosine_similarity(tfidf_mat[0:1], tfidf_mat[1:2])[0][0])
            return float(min(1.0, max(0.0, cos_sim)))
        except Exception:
            return 0.0

    def compute_batch_similarities(
        self,
        target_jd: str,
        candidate_texts: Sequence[str],
    ) -> list[float]:
        """
        Fits vectorizer across all candidate texts and the target JD simultaneously,
        then evaluates all cosine similarities against target_jd in a single step.
        """
        clean_jd = sanitize_text(target_jd)
        clean_candidates = [sanitize_text(c) for c in candidate_texts]

        if not clean_jd:
            return [0.0] * len(candidate_texts)

        # Fit on candidate texts + JD
        all_docs = [clean_jd] + clean_candidates
        self.fit(all_docs)

        matrix = self.vectorizer.transform(all_docs)
        jd_vector = matrix[0:1]
        cand_vectors = matrix[1:]

        sims = cosine_similarity(cand_vectors, jd_vector).flatten()
        return [float(min(1.0, max(0.0, s))) for s in sims]
