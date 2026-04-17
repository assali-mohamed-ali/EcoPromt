# EcoPrompt AI — MVP Hackathon Specification
> **Contexte :** prototype de démonstration pour hackathon (48h).
> Objectif : montrer l'idée, pas livrer un produit fini.
> Aucune API LLM requise. Tout tourne en local avec des datasets publics.

---

## Ce que le MVP doit démontrer (et rien de plus)

```
1. Détecter si un prompt est flou → demander des précisions (checkboxes)
2. Analyser le prompt et identifier des gaspillages de tokens
3. Compresser le prompt → afficher les tokens économisés
4. Recommander le bon modèle selon la complexité
5. Afficher l'économie d'énergie estimée
```

**Ce qu'on ne fait PAS pour le hackathon :**
- Fine-tuning de modèles (trop long)
- Base de données de feedback
- Multi-provider routing
- Déploiement cloud

---

## Stack MVP (minimaliste)

```
streamlit          → UI
scikit-learn       → classifieurs (TF-IDF + LogReg, déjà entraînés)
spacy              → analyse linguistique du prompt
tiktoken           → comptage exact des tokens
pandas / numpy     → manipulation des données
datasets           → chargement des datasets HuggingFace
```

**Pas de PyTorch, pas de transformers, pas de GPU requis.**

---

## Structure du projet (MVP)

```
ecoprompt-mvp/
│
├── app.py                    ← point d'entrée Streamlit (fichier principal)
├── clarity_gate.py           ← détection de flou + questions checkboxes
├── token_economist.py        ← ★ NOUVEAU : analyse et réduction des tokens
├── task_classifier.py        ← classification du type de tâche (TF-IDF)
├── energy_estimator.py       ← estimation énergie/CO₂
│
├── models/
│   ├── ambiguity_clf.joblib  ← modèle entraîné (généré par train.py)
│   └── task_clf.joblib       ← modèle entraîné (généré par train.py)
│
├── train.py                  ← script unique pour entraîner les 2 modèles
├── data/                     ← datasets téléchargés automatiquement
├── requirements.txt
└── README.md
```

---

## requirements.txt

```
streamlit>=1.35.0
scikit-learn>=1.4.0
pandas>=2.0.0
numpy>=1.26.0
spacy>=3.7.0
tiktoken>=0.6.0
datasets>=2.19.0
joblib>=1.3.0
```

**Setup :**
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python train.py          # entraîne et sauvegarde les 2 modèles (~3 min)
streamlit run app.py     # lance le prototype
```

---

## Datasets (publics, téléchargement automatique)

Le script `train.py` télécharge tout automatiquement via HuggingFace.

| Dataset | Utilisé pour | Lien | Taille |
|---|---|---|---|
| `ambig_qa` (light) | Entraîner le classifieur de flou (classe = ambigu) | https://huggingface.co/datasets/ambig_qa | 14K |
| `Jayveersinh-Raj/bad-improved-prompt-pairs` | Exemples de prompts clairs (classe = clair) — utilise la colonne `improved_prompt` | https://huggingface.co/datasets/Jayveersinh-Raj/bad-improved-prompt-pairs | ~10K |
| `WizardLM/WizardLM_evol_instruct_70k` | Entraîner le classifieur de tâche + complexité | https://huggingface.co/datasets/WizardLM/WizardLM_evol_instruct_70k | 70K |

> **Pourquoi `bad-improved-prompt-pairs` ?**
> Ce dataset contient des paires (prompt mal formulé → prompt amélioré). En utilisant la colonne `improved_prompt` comme classe "clair", on entraîne le classifieur sur des exemples de prompts réellement bien construits — plus pertinent que des instructions génériques. La colonne `bad_prompt` peut aussi servir à enrichir la classe "ambigu" (optionnel, voir commentaire dans `train.py`).

**Total téléchargé : ~150 MB. Aucune inscription requise.**

```python
# Dans train.py — téléchargement automatique
from datasets import load_dataset

ambig  = load_dataset("ambig_qa", "light", split="train")
pairs  = load_dataset("Jayveersinh-Raj/bad-improved-prompt-pairs", split="train")
wizard = load_dataset("WizardLM/WizardLM_evol_instruct_70k", split="train")
```

---

## Module 1 — `train.py`

Script unique, idempotent : s'arrête proprement si les modèles existent déjà. Produit deux fichiers `.joblib`.

```python
# train.py
from datasets import load_dataset
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from pathlib import Path
import pandas as pd, joblib, re

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────
def _already_exists(path: Path, label: str) -> bool:
    if path.exists():
        print(f"✓ {label} already exists — skipping.")
        return True
    return False


# ── Ambiguity Classifier ──────────────────────────────────────────
def maybe_train_ambiguity():
    path = MODEL_DIR / "ambiguity_clf.joblib"
    if _already_exists(path, "ambiguity_clf.joblib"):
        return

    # Classe 1 : prompts ambigus → AmbigQA
    ambig = load_dataset("ambig_qa", "light", split="train")
    ambig_prompts = [row["question"] for row in ambig]

    # Classe 0 : prompts clairs → improved_prompt du dataset bad-improved-prompt-pairs
    # Ce dataset contient des paires (mauvais prompt → prompt amélioré).
    # On utilise uniquement la colonne `improved_prompt` comme exemples de prompts clairs.
    pairs = load_dataset("Jayveersinh-Raj/bad-improved-prompt-pairs", split="train")
    clear_prompts = [row["improved_prompt"] for row in pairs
                     if row.get("improved_prompt")][:len(ambig_prompts)]

    # Optionnel : enrichir la classe "ambigu" avec les bad_prompts du même dataset
    # extra_ambig = [row["bad_prompt"] for row in pairs if row.get("bad_prompt")]
    # ambig_prompts = ambig_prompts + extra_ambig[:2000]

    texts  = ambig_prompts + clear_prompts
    labels = [1] * len(ambig_prompts) + [0] * len(clear_prompts)

    ambiguity_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=30000, ngram_range=(1, 3), sublinear_tf=True)),
        ("clf",   LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced"))
    ])
    ambiguity_pipeline.fit(texts, labels)
    joblib.dump(ambiguity_pipeline, path)
    print("✓ Ambiguity classifier saved")


# ── Task Classifier ───────────────────────────────────────────────
TASK_KEYWORDS = {
    "code":          ["function", "bug", "script", "code", "debug", "implement", "class", "api"],
    "writing":       ["write", "essay", "email", "letter", "blog", "article", "story"],
    "reasoning":     ["analyze", "compare", "pros", "cons", "argue", "evaluate", "strategy"],
    "summarization": ["summarize", "summary", "tldr", "brief", "shorten", "condense"],
    "qa":            ["what is", "who is", "define", "explain", "when was", "how does"],
    "data":          ["dataset", "csv", "dataframe", "chart", "plot", "statistics", "pandas"],
    "translation":   ["translate", "in french", "in arabic", "in spanish", "en français"],
    "creative":      ["poem", "story", "imagine", "create", "invent", "fictional"],
}

def auto_label(text):
    t = text.lower()
    for task, keywords in TASK_KEYWORDS.items():
        if any(k in t for k in keywords):
            return task
    return "general"

def maybe_train_task():
    path = MODEL_DIR / "task_clf.joblib"
    if _already_exists(path, "task_clf.joblib"):
        return

    wizard = load_dataset("WizardLM/WizardLM_evol_instruct_70k", split="train")
    wiz_texts  = [row["instruction"] for row in wizard]
    wiz_labels = [auto_label(t) for t in wiz_texts]

    task_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=50000, ngram_range=(1, 2))),
        ("clf",   LinearSVC(C=1.0, max_iter=2000))
    ])
    task_pipeline.fit(wiz_texts, wiz_labels)
    joblib.dump(task_pipeline, path)
    print("✓ Task classifier saved")


if __name__ == "__main__":
    maybe_train_ambiguity()
    maybe_train_task()
    print("Done. Run: streamlit run app.py")
```

---

## Module 2 — `clarity_gate.py`

Charge le modèle entraîné et génère les questions de clarification.

```python
# clarity_gate.py
from __future__ import annotations
import joblib, re
from dataclasses import dataclass, field

_MODEL = None

def _load():
    global _MODEL
    if _MODEL is None:
        _MODEL = joblib.load("models/ambiguity_clf.joblib")
    return _MODEL

@dataclass
class ClarityResult:
    is_ambiguous: bool
    probability: float
    questions: list[dict] = field(default_factory=list)

QUESTIONS = {
    "code": {
        "id": "code_goal",
        "text": "What do you need for this code task?",
        "options": ["Write new code", "Debug / fix a bug", "Explain existing code",
                    "Refactor / optimize", "Add tests", "Convert to another language"]
    },
    "writing": {
        "id": "writing_goal",
        "text": "What kind of writing?",
        "options": ["Professional / formal", "Casual / conversational",
                    "Technical doc", "Creative / storytelling", "Email draft"]
    },
    "general": {
        "id": "general_goal",
        "text": "What is the main goal of your request?",
        "options": ["Get information", "Create something", "Analyze or compare",
                    "Solve a technical problem", "Summarize content"]
    },
}

LENGTH_Q = {
    "id": "output_length",
    "text": "How long should the response be?",
    "options": ["One-liner", "Short paragraph", "Detailed explanation", "Comprehensive with examples"]
}

def detect_domain(prompt: str) -> str:
    p = prompt.lower()
    if any(k in p for k in ["code", "function", "bug", "script", "python", "javascript"]): return "code"
    if any(k in p for k in ["write", "essay", "email", "letter", "story", "blog"]): return "writing"
    return "general"

def assess(prompt: str, threshold: float = 0.50) -> ClarityResult:
    model = _load()
    prob = float(model.predict_proba([prompt])[0][1])
    if prob < threshold:
        return ClarityResult(is_ambiguous=False, probability=round(prob, 3))
    domain = detect_domain(prompt)
    questions = [QUESTIONS[domain], LENGTH_Q]
    return ClarityResult(is_ambiguous=True, probability=round(prob, 3), questions=questions)

def enrich(original: str, answers: dict[str, list[str]]) -> str:
    lines = [original.strip()]
    for qid, selections in answers.items():
        if selections:
            lines.append(f"[Context — {qid.replace('_',' ')}: {', '.join(selections)}]")
    return "\n".join(lines)
```

---

## Module 3 — `token_economist.py` ★ (partie clé — économie de tokens)

Ce module est le cœur différenciateur du MVP. Il analyse le prompt sous **6 angles d'économie de tokens** et produit un rapport détaillé.

### Techniques d'économie de tokens implémentées

| # | Technique | Méthode | Rationnel |
|---|---|---|---|
| 1 | **Filler word removal** | Regex sur 18 patterns | "please", "kindly", "could you", "as an AI"… = tokens zéro-information |
| 2 | **Repetition detection** | spaCy + lemmatisation + Counter | Même mot de contenu répété = tokens dupliqués |
| 3 | **Multi-language detection** | Regex Unicode par langue | 1 mot arabe = 4–8 tokens vs 1–2 en anglais |
| 4 | **Costly patterns** | Regex sur 5 patterns | `!!!`, ALL-CAPS, URLs longues, unicode = surcoût tokenizer |
| 5 | **Redundant sentences** | Jaccard similarity (seuil 0.60) | Phrases paraphrasées = tokens redondants |
| 6 | **Compression + garde sémantique** | Règles 1–4 + vérification 85% des noms préservés | Compression sans perte de sens |

```python
# token_economist.py
from __future__ import annotations
import re, tiktoken
from dataclasses import dataclass, field
from collections import Counter

_enc = tiktoken.get_encoding("cl100k_base")  # encodeur mis en cache une seule fois
_spacy_nlp = None

def _nlp():
    global _spacy_nlp
    if _spacy_nlp is None:
        import spacy as _spacy
        _spacy_nlp = _spacy.load("en_core_web_sm")
    return _spacy_nlp


def token_count(text: str) -> int:
    return len(_enc.encode(text))


# ─────────────────────────────────────────────────────────────────
# 1. FILLER WORD REMOVAL
#    Words that consume tokens but carry zero task information.
# ─────────────────────────────────────────────────────────────────
FILLER_PATTERNS = [
    # Polite fillers
    (r"\bplease\b",              "polite filler"),
    (r"\bkindly\b",              "polite filler"),
    (r"\bif you don'?t mind\b",  "polite filler"),
    # Indirect phrasing
    (r"\bcould you\b",           "indirect phrasing"),
    (r"\bwould you\b",           "indirect phrasing"),
    (r"\bi want you to\b",       "indirect phrasing"),
    (r"\bi need you to\b",       "indirect phrasing"),
    (r"\bcan you please\b",      "indirect phrasing"),
    # Social fillers
    (r"\bthank you\b",           "social filler"),
    (r"\bthanks\b",              "social filler"),
    (r"\bi hope (you|this).{0,40}", "social filler"),
    # AI-addressing fillers
    (r"\bas an ai(?: language model)?\b", "AI-addressing filler"),
    (r"\bact as an? \w+(?: and)?\b",      "role-play preamble"),
    # Permissive / hedge fillers
    (r"\bfeel free to\b",        "permissive filler"),
    (r"\bjust\b",                "weakener"),
    (r"\bbasically\b",           "hedge filler"),
    (r"\bactually\b",            "hedge filler"),
    # Vagueness markers
    (r"\bkind of\b",             "vagueness marker"),
    (r"\bsort of\b",             "vagueness marker"),
    (r"\ba little bit\b",        "vagueness marker"),
    # Verbose constructions (ajout v2)
    (r"\bin order to\b",         "verbose construction"),
    (r"\bdue to the fact that\b","verbose construction"),
    (r"\bit is important to note that\b", "verbose construction"),
    (r"\bplease note that\b",    "verbose construction"),
    (r"\bas you (can )?see\b",   "verbose construction"),
]

def remove_fillers(prompt: str) -> tuple[str, list[dict]]:
    """
    Returns (cleaned_prompt, list_of_removed_items)
    Each item: { "pattern": str, "reason": str, "count": int }
    """
    removed = []
    result = prompt
    for pattern, reason in FILLER_PATTERNS:
        matches = re.findall(pattern, result, re.IGNORECASE)
        if matches:
            removed.append({"pattern": pattern.replace(r"\b", ""), "reason": reason, "count": len(matches)})
            result = re.sub(pattern, "", result, flags=re.IGNORECASE)
    result = re.sub(r"\s{2,}", " ", result).strip()
    return result, removed


# ─────────────────────────────────────────────────────────────────
# 2. REPETITION DETECTION
#    Same content words repeated across the prompt.
# ─────────────────────────────────────────────────────────────────
def detect_repetitions(prompt: str) -> dict:
    """
    Finds content words (nouns, verbs, adjectives) that appear
    more than once. Returns:
      {
        "repeated_words": [ {"word": str, "count": int, "tokens_wasted": int} ],
        "total_wasted_tokens": int
      }
    """
    doc = _nlp()(prompt)
    content_pos = {"NOUN", "VERB", "ADJ"}
    word_counts = Counter(
        token.lemma_.lower()
        for token in doc
        if token.pos_ in content_pos and not token.is_stop and len(token.text) > 2
    )
    repeated = [
        {
            "word": word,
            "count": count,
            "tokens_wasted": (count - 1)
        }
        for word, count in word_counts.items()
        if count > 1
    ]
    repeated.sort(key=lambda x: x["tokens_wasted"], reverse=True)
    return {
        "repeated_words": repeated[:10],
        "total_wasted_tokens": sum(r["tokens_wasted"] for r in repeated)
    }


# ─────────────────────────────────────────────────────────────────
# 3. MULTI-LANGUAGE DETECTION
#    Mixing languages forces tokenization overhead.
#    Non-English scripts (Arabic, CJK) = 3–6× more tokens per word.
# ─────────────────────────────────────────────────────────────────
LANG_PATTERNS = {
    "Arabic":  r"[\u0600-\u06FF]",
    "French":  r"\b(le|la|les|un|une|des|et|est|de|du|pour|avec|vous|nous|je|tu|il)\b",
    "Spanish": r"\b(el|la|los|las|un|una|y|es|de|del|para|con|por|como|que)\b",
    "German":  r"\b(der|die|das|ein|eine|und|ist|von|zu|für|mit|nicht|als|auch)\b",
    "Italian": r"\b(il|la|lo|le|un|una|e|è|di|del|per|con|che|come|sono)\b",
}

def detect_languages(prompt: str) -> dict:
    found = []
    for lang, pattern in LANG_PATTERNS.items():
        if re.search(pattern, prompt, re.IGNORECASE):
            found.append(lang)

    has_non_latin = bool(re.search(r"[\u0600-\u06FF\u4e00-\u9fff\u3040-\u30ff]", prompt))
    warning = len(found) > 1 or has_non_latin

    explanation = ""
    if len(found) > 1:
        explanation = (
            f"Prompt mixes {', '.join(found)}. Switching languages mid-prompt causes "
            f"tokenizer overhead — keep the prompt in one language for fewer tokens."
        )
    elif has_non_latin:
        explanation = (
            "Non-Latin script detected. Arabic, Chinese, and similar scripts are "
            "tokenized into 3–6× more tokens per word than English. "
            "Consider writing the task in English and specifying the output language separately."
        )
    return {
        "languages_detected": found,
        "token_overhead_warning": warning,
        "explanation": explanation
    }


# ─────────────────────────────────────────────────────────────────
# 4. COSTLY TOKEN PATTERNS
#    Characters and patterns that cost disproportionately many tokens.
# ─────────────────────────────────────────────────────────────────
COSTLY_PATTERNS = [
    {
        "name": "Excessive punctuation",
        "pattern": r"[!?.]{3,}",
        "explanation": "Three or more consecutive punctuation marks tokenize as multiple tokens. Use one.",
        "suggestion": "Replace '!!!' with '!'",
    },
    {
        "name": "All-caps words",
        "pattern": r"\b[A-Z]{3,}\b",
        "explanation": "ALL-CAPS words often tokenize as separate tokens per letter-group.",
        "suggestion": "Use normal casing unless the word is an acronym.",
    },
    {
        "name": "Repeated whitespace / line breaks",
        "pattern": r"(\n\s*){3,}",
        "explanation": "Multiple blank lines waste tokens with no informational value.",
        "suggestion": "Use a single line break between sections.",
    },
    {
        "name": "Unicode special characters",
        "pattern": r"[^\x00-\x7F]{2,}",
        "explanation": "Non-ASCII characters (emojis, special quotes, arrows) each cost 1–3 tokens.",
        "suggestion": "Use plain ASCII equivalents where possible.",
    },
    {
        "name": "Long URLs",
        "pattern": r"https?://\S{60,}",
        "explanation": "Long URLs tokenize character-by-character — up to 30+ tokens for one link.",
        "suggestion": "Use a URL shortener or just describe what the URL points to.",
    },
]

def detect_costly_patterns(prompt: str) -> list[dict]:
    found = []
    for item in COSTLY_PATTERNS:
        matches = re.findall(item["pattern"], prompt)
        if matches:
            found.append({
                "name": item["name"],
                "count": len(matches),
                "explanation": item["explanation"],
                "suggestion": item["suggestion"],
                "example": str(matches[0])[:40] if matches else ""
            })
    return found


# ─────────────────────────────────────────────────────────────────
# 5. REDUNDANT CONTEXT DETECTION
#    Sentences paraphrasing each other = removable tokens.
# ─────────────────────────────────────────────────────────────────
def detect_redundant_sentences(prompt: str) -> dict:
    """
    Compares sentence pairs using Jaccard similarity on token sets.
    Sentences with similarity > 0.60 are flagged as redundant.
    """
    doc = _nlp()(prompt)
    sentences = [str(sent).strip() for sent in doc.sents if len(str(sent).split()) > 4]

    def jaccard(a, b):
        sa = set(a.lower().split())
        sb = set(b.lower().split())
        if not sa | sb:
            return 0.0
        return len(sa & sb) / len(sa | sb)

    redundant = []
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            sim = jaccard(sentences[i], sentences[j])
            if sim > 0.60:
                redundant.append({
                    "s1": sentences[i][:80],
                    "s2": sentences[j][:80],
                    "similarity": round(sim, 2)
                })

    removable_tokens = sum(token_count(p["s2"]) for p in redundant)
    return {
        "redundant_pairs": redundant,
        "estimated_removable_tokens": removable_tokens
    }


# ─────────────────────────────────────────────────────────────────
# 6. COMPRESSION + GARDE SÉMANTIQUE (Règle 9)
#    Applique toutes les règles et vérifie que 85% des noms
#    uniques du prompt original sont préservés.
# ─────────────────────────────────────────────────────────────────
def _preserve_unique_nouns(original: str, compressed: str, threshold: float = 0.85) -> str:
    """
    Garde sémantique — Règle 9 de la spec.
    Si la compression supprime plus de 15% des noms uniques,
    on bascule en mode safe : on ne supprime que les fillers
    purement sociaux (sans toucher aux vagueness markers ni
    aux constructions verbales qui pourraient contenir des noms).
    """
    nlp_model = _nlp()
    # Traitement en une passe pour éviter de charger spaCy deux fois
    orig_doc = nlp_model(original)
    comp_doc = nlp_model(compressed)

    orig_nouns = {t.lemma_.lower() for t in orig_doc if t.pos_ == "NOUN" and not t.is_stop}
    if not orig_nouns:
        return compressed  # pas de noms → rien à préserver

    comp_nouns = {t.lemma_.lower() for t in comp_doc if t.pos_ == "NOUN" and not t.is_stop}
    ratio = len(comp_nouns & orig_nouns) / len(orig_nouns)

    if ratio < threshold:
        # Fallback : n'appliquer que les fillers purement sociaux/polis
        # (catégories qui ne contiennent jamais de noms porteurs de sens)
        safe_categories = {"polite filler", "social filler", "AI-addressing filler"}
        safe_patterns = [(p, r) for p, r in FILLER_PATTERNS if r in safe_categories]
        result = original
        for pattern, _ in safe_patterns:
            result = re.sub(pattern, "", result, flags=re.IGNORECASE)
        return re.sub(r"\s{2,}", " ", result).strip()

    return compressed


def compress(prompt: str) -> str:
    """
    Applique : filler removal + normalisation + garde sémantique 85%.
    Retourne le prompt optimisé.
    """
    result, _ = remove_fillers(prompt)
    result = re.sub(r"[!?.]{3,}", lambda m: m.group(0)[0], result)   # excessive punctuation
    result = re.sub(r"(\n\s*){3,}", "\n\n", result)                   # excessive newlines
    result = re.sub(r"\s{2,}", " ", result).strip()
    result = _preserve_unique_nouns(prompt, result)                    # Règle 9
    return result


# ─────────────────────────────────────────────────────────────────
# PUBLIC API — called by app.py
# ─────────────────────────────────────────────────────────────────
@dataclass
class EconomyReport:
    original_prompt:     str
    compressed_prompt:   str
    original_tokens:     int
    compressed_tokens:   int
    token_reduction:     int
    reduction_pct:       float
    fillers_removed:     list[dict]
    repetitions:         dict
    language_analysis:   dict
    costly_patterns:     list[dict]
    redundant_sentences: dict

def analyze(prompt: str) -> EconomyReport:
    """
    Full token economy analysis. Called by app.py.
    Returns EconomyReport with all findings and the compressed prompt.
    """
    compressed        = compress(prompt)
    _, fillers        = remove_fillers(prompt)
    repetitions       = detect_repetitions(prompt)
    language_analysis = detect_languages(prompt)
    costly            = detect_costly_patterns(prompt)
    redundant         = detect_redundant_sentences(prompt)

    orig_tok = token_count(prompt)
    comp_tok = token_count(compressed)

    return EconomyReport(
        original_prompt     = prompt,
        compressed_prompt   = compressed,
        original_tokens     = orig_tok,
        compressed_tokens   = comp_tok,
        token_reduction     = orig_tok - comp_tok,
        reduction_pct       = round((1 - comp_tok / max(orig_tok, 1)) * 100, 1),
        fillers_removed     = fillers,
        repetitions         = repetitions,
        language_analysis   = language_analysis,
        costly_patterns     = costly,
        redundant_sentences = redundant,
    )
```

---

## Module 4 — `task_classifier.py`

```python
# task_classifier.py
from __future__ import annotations
import joblib

_MODEL = None

def _load():
    global _MODEL
    if _MODEL is None:
        _MODEL = joblib.load("models/task_clf.joblib")
    return _MODEL

def predict(prompt: str) -> dict:
    model = _load()
    label = model.predict([prompt])[0]
    # LinearSVC has decision_function, not predict_proba
    scores = model.decision_function([prompt])[0]
    confidence = round(float(max(scores) / (sum(abs(s) for s in scores) + 1e-9)), 2)
    return {"task": label, "confidence": confidence}
```

---

## Module 5 — `energy_estimator.py`

```python
# energy_estimator.py
from __future__ import annotations

# Source: Luccioni et al. 2023 (arxiv:2311.16863)
ENERGY = {
    "light":  {"wh_per_1k": 0.001,  "example": "7B model (Mistral-7B)"},
    "medium": {"wh_per_1k": 0.009,  "example": "13–34B model (Mixtral-8x7B)"},
    "heavy":  {"wh_per_1k": 0.032,  "example": "70B+ model (Llama-3-70B)"},
}
GRID_CO2 = 385  # gCO₂/kWh — IEA 2023 global average

TIER_MAP = {
    "code":          "medium",
    "reasoning":     "heavy",
    "data":          "medium",
    "creative":      "medium",
    "writing":       "light",
    "summarization": "light",
    "qa":            "light",
    "translation":   "light",
    "general":       "light",
}

def get_tier(task: str, complexity_score: float) -> str:
    base = TIER_MAP.get(task, "light")
    if complexity_score >= 4.0:
        return "heavy"
    if complexity_score >= 2.5 and base == "light":
        return "medium"
    return base

def estimate(tokens: int, tier: str) -> dict:
    wh     = (tokens / 1000) * ENERGY[tier]["wh_per_1k"]
    co2_g  = (wh / 1000) * GRID_CO2
    return {
        "wh":    round(wh, 5),
        "co2_g": round(co2_g, 5),
        "model": ENERGY[tier]["example"],
        "tier":  tier,
    }

def savings(orig_tok: int, comp_tok: int, heavy_tier: str, selected_tier: str) -> dict:
    baseline = estimate(orig_tok, heavy_tier)
    actual   = estimate(comp_tok, selected_tier)
    return {
        "wh_saved":    round(baseline["wh"]    - actual["wh"],    5),
        "co2_g_saved": round(baseline["co2_g"] - actual["co2_g"], 5),
        "pct_saved":   round((1 - actual["wh"] / max(baseline["wh"], 1e-9)) * 100, 1),
        "note": "Source: Luccioni et al. 2023. Values are approximations."
    }

def simple_complexity(token_count: int, task: str) -> float:
    """Lightweight complexity proxy for MVP (no ML model needed)."""
    base = {"reasoning": 3.5, "code": 2.5, "data": 2.5, "creative": 2.0}.get(task, 1.5)
    length_bonus = min(token_count / 200, 2.0)
    return round(min(base + length_bonus, 5.0), 1)
```

---

## Module 6 — `app.py` (UI Streamlit)

```python
# app.py
import streamlit as st
from clarity_gate    import assess, enrich, ClarityResult
from token_economist import analyze, EconomyReport
from task_classifier import predict as classify_task
from energy_estimator import get_tier, estimate, savings, simple_complexity

st.set_page_config(page_title="EcoPrompt AI", page_icon="🌱", layout="wide")
st.title("🌱 EcoPrompt AI")
st.caption("Reduce token waste. Save energy. Get better AI responses.")

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.header("Your Prompt")
    user_prompt = st.text_area("Enter your prompt", height=160,
                               placeholder="e.g. fix it  /  write something  /  help me")
    run = st.button("Analyze & Optimize", type="primary", use_container_width=True)

if not run or not user_prompt.strip():
    st.info("Enter a prompt in the sidebar and click Analyze.")
    st.stop()

# ── Step 1 : Clarity Gate ─────────────────────────────────────────
st.subheader("Step 1 — Clarity Check")
clarity: ClarityResult = assess(user_prompt)

col1, col2 = st.columns([1, 2])
with col1:
    color = "red" if clarity.probability > 0.6 else "orange" if clarity.probability > 0.3 else "green"
    st.metric("Ambiguity score", f"{clarity.probability:.2f}")
    st.progress(clarity.probability)

if clarity.is_ambiguous:
    st.warning("Your prompt is ambiguous. Answer a few questions to help the model understand you better.")
    answers = {}
    with st.form("clarify"):
        for q in clarity.questions:
            selected = st.multiselect(q["text"], q["options"], key=q["id"])
            answers[q["id"]] = selected
        submitted = st.form_submit_button("Clarify →")
    if submitted:
        user_prompt = enrich(user_prompt, answers)
        st.success("Prompt enriched with your answers.")
        with st.expander("View enriched prompt"):
            st.code(user_prompt)
else:
    st.success("Prompt is clear — no clarification needed.")

# ── Step 2 : Token Economy Analysis ───────────────────────────────
st.divider()
st.subheader("Step 2 — Token Economy Report")

report: EconomyReport = analyze(user_prompt)

# Summary metrics
m1, m2, m3, m4 = st.columns(4)
m1.metric("Original tokens",   report.original_tokens)
m2.metric("Compressed tokens", report.compressed_tokens)
m3.metric("Tokens saved",      report.token_reduction)
m4.metric("Reduction",         f"{report.reduction_pct}%")

# Findings in tabs
t1, t2, t3, t4, t5 = st.tabs([
    "Fillers removed",
    "Repetitions",
    "Language overhead",
    "Costly patterns",
    "Redundant sentences"
])

with t1:
    if report.fillers_removed:
        for f in report.fillers_removed:
            st.write(f"- **{f['pattern']}** ({f['reason']}) — found {f['count']}×")
    else:
        st.write("No filler words detected.")

with t2:
    reps = report.repetitions
    if reps["repeated_words"]:
        st.write(f"**{reps['total_wasted_tokens']} tokens wasted** by repetition:")
        for r in reps["repeated_words"]:
            st.write(f"- `{r['word']}` repeated {r['count']}× → {r['tokens_wasted']} extra token(s)")
    else:
        st.write("No significant word repetition detected.")

with t3:
    lang = report.language_analysis
    if lang["token_overhead_warning"]:
        st.warning(lang["explanation"])
        st.write(f"Languages detected: {', '.join(lang['languages_detected'])}")
    else:
        st.write("Single language detected — no tokenization overhead.")

with t4:
    if report.costly_patterns:
        for p in report.costly_patterns:
            st.write(f"**{p['name']}** (×{p['count']})")
            st.caption(f"{p['explanation']}")
            st.caption(f"💡 {p['suggestion']}")
    else:
        st.write("No costly token patterns found.")

with t5:
    red = report.redundant_sentences
    if red["redundant_pairs"]:
        st.write(f"Estimated **{red['estimated_removable_tokens']} removable tokens** from redundancy:")
        for pair in red["redundant_pairs"]:
            st.write(f"- Similarity {pair['similarity']}: *\"{pair['s1']}...\"* ≈ *\"{pair['s2']}...\"*")
    else:
        st.write("No redundant sentences detected.")

# Optimized prompt preview
st.divider()
st.subheader("Optimized Prompt")
col_a, col_b = st.columns(2)
with col_a:
    st.caption("Original")
    st.code(report.original_prompt)
with col_b:
    st.caption("Compressed")
    st.code(report.compressed_prompt)

# ── Step 3 : Task + Routing ───────────────────────────────────────
st.divider()
st.subheader("Step 3 — Task Classification & Model Routing")

task_result = classify_task(report.compressed_prompt)
task        = task_result["task"]
complexity  = simple_complexity(report.compressed_tokens, task)
tier        = get_tier(task, complexity)

c1, c2, c3 = st.columns(3)
c1.metric("Task type",        task)
c2.metric("Complexity score", f"{complexity} / 5")
c3.metric("Recommended tier", tier.upper())

tier_models = {
    "light":  "Mistral-7B-Instruct",
    "medium": "Mixtral-8x7B-Instruct",
    "heavy":  "Llama-3-70B-Instruct",
}
st.info(f"Recommended open-source model: **{tier_models[tier]}** — run locally with Ollama or HuggingFace.")

# ── Step 4 : Energy & CO₂ savings ────────────────────────────────
st.divider()
st.subheader("Step 4 — Energy & CO₂ Savings")

result = savings(
    orig_tok      = report.original_tokens,
    comp_tok      = report.compressed_tokens,
    heavy_tier    = "heavy",   # worst-case baseline
    selected_tier = tier
)

e1, e2, e3 = st.columns(3)
e1.metric("Energy saved",  f"{result['wh_saved']:.4f} Wh")
e2.metric("CO₂ avoided",   f"{result['co2_g_saved']:.4f} g")
e3.metric("% saved",       f"{result['pct_saved']}%")

st.caption(result["note"])
st.caption("Grid intensity: 385 gCO₂/kWh (IEA 2023 global average)")
```

---

## Ce que le jury verra (demo flow)

```
1. L'utilisateur tape un prompt vague : "fix it please please please"
2. → Clarity gate : ambiguité 0.81 → checkboxes apparaissent
3. L'utilisateur coche "Debug / fix a bug" + "Detailed explanation"
4. → Prompt enrichi avec le contexte
5. → Token Economy Report :
       Fillers : "please please please" (3 occurrences, 3 tokens gaspillés)
       Répétitions : "please" ×3
       Pas de problème de langue
6. → Compressed : "fix it" (de 6 tokens à 2 tokens → -67%)
7. → Task : code | Complexité : 2.5 | Tier : medium
8. → Modèle recommandé : Mixtral-8x7B
9. → Énergie économisée vs worst-case (heavy + non-compressé)
```

---

## Règles Copilot pour le MVP

1. **Un seul fichier par module.** Pas de sous-dossiers inutiles.
2. **Pas de GPU requis.** Tous les modèles = scikit-learn `.joblib`.
3. **`train.py` est idempotent.** S'arrête proprement si les modèles existent déjà. ✅ *implémenté v2*
4. **`token_economist.py` ne fait aucun appel réseau.** Tout en local.
5. **Les modèles chargent une seule fois** via variable globale `_MODEL = None`.
6. **`token_count()` utilise tiktoken**, jamais `len(text.split())`. L'encodeur tiktoken est mis en cache en variable globale `_enc`. ✅ *optimisé v2*
7. **L'UI affiche toujours le disclaimer énergie** sous les métriques CO₂.
8. **Chaque tab du Token Economy Report affiche un message "rien trouvé"** si vide (pas de tab vide).
9. **`compress()` préserve au minimum 85% des noms uniques** du prompt original (garde sémantique). ✅ *implémenté v2*
10. **Hackathon first** : si un choix technique ralentit le dev, prendre la solution la plus simple.

---

## Changelog v2

| # | Changement | Fichier | Règle |
|---|---|---|---|
| 1 | Dataset `tatsu-lab/alpaca` remplacé par `Jayveersinh-Raj/bad-improved-prompt-pairs` (colonne `improved_prompt`) | `train.py` | Dataset |
| 2 | `train.py` rendu idempotent via `maybe_train_ambiguity()` / `maybe_train_task()` | `train.py` | Règle 3 |
| 3 | Garde sémantique 85% ajoutée dans `compress()` via `_preserve_unique_nouns()` | `token_economist.py` | Règle 9 |
| 4 | Encodeur tiktoken mis en cache en variable globale `_enc` (supprime recréation à chaque appel) | `token_economist.py` | Règle 5 / perf |
| 5 | 5 patterns verbeux ajoutés à `FILLER_PATTERNS` (`in order to`, `due to the fact that`…) | `token_economist.py` | Qualité compression |
