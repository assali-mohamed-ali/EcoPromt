from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

import tiktoken

_enc = tiktoken.get_encoding("cl100k_base")
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
# ─────────────────────────────────────────────────────────────────
FILLER_PATTERNS = [
    # Polite fillers
    (r"\bplease\b", "polite filler"),
    (r"\bkindly\b", "polite filler"),
    (r"\bif you don'?t mind\b", "polite filler"),
    # Indirect phrasing
    (r"\bcould you\b", "indirect phrasing"),
    (r"\bwould you\b", "indirect phrasing"),
    (r"\bi want you to\b", "indirect phrasing"),
    (r"\bi need you to\b", "indirect phrasing"),
    (r"\bcan you please\b", "indirect phrasing"),
    # Social fillers
    (r"\bthank you\b", "social filler"),
    (r"\bthanks\b", "social filler"),
    (r"\bi hope (you|this).{0,40}", "social filler"),
    # AI-addressing fillers
    (r"\bas an ai(?: language model)?\b", "AI-addressing filler"),
    (r"\bact as an? \w+(?: and)?\b", "role-play preamble"),
    # Permissive / hedge fillers
    (r"\bfeel free to\b", "permissive filler"),
    (r"\bjust\b", "weakener"),
    (r"\bbasically\b", "hedge filler"),
    (r"\bactually\b", "hedge filler"),
    # Vagueness markers
    (r"\bkind of\b", "vagueness marker"),
    (r"\bsort of\b", "vagueness marker"),
    (r"\ba little bit\b", "vagueness marker"),
    # Verbose constructions
    (r"\bin order to\b", "verbose construction"),
    (r"\bdue to the fact that\b", "verbose construction"),
    (r"\bit is important to note that\b", "verbose construction"),
    (r"\bplease note that\b", "verbose construction"),
    (r"\bas you (can )?see\b", "verbose construction"),
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
        {"word": word, "count": count, "tokens_wasted": (count - 1)}
        for word, count in word_counts.items()
        if count > 1
    ]
    repeated.sort(key=lambda x: x["tokens_wasted"], reverse=True)
    return {
        "repeated_words": repeated[:10],
        "total_wasted_tokens": sum(r["tokens_wasted"] for r in repeated),
    }


# ─────────────────────────────────────────────────────────────────
# 3. MULTI-LANGUAGE DETECTION
# ─────────────────────────────────────────────────────────────────
LANG_PATTERNS = {
    "Arabic": r"[\u0600-\u06FF]",
    "French": r"\b(le|la|les|un|une|des|et|est|de|du|pour|avec|vous|nous|je|tu|il)\b",
    "Spanish": r"\b(el|la|los|las|un|una|y|es|de|del|para|con|por|como|que)\b",
    "German": r"\b(der|die|das|ein|eine|und|ist|von|zu|für|mit|nicht|als|auch)\b",
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
    return {"languages_detected": found, "token_overhead_warning": warning, "explanation": explanation}


# ─────────────────────────────────────────────────────────────────
# 4. COSTLY TOKEN PATTERNS
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
            found.append(
                {
                    "name": item["name"],
                    "count": len(matches),
                    "explanation": item["explanation"],
                    "suggestion": item["suggestion"],
                    "example": str(matches[0])[:40] if matches else "",
                }
            )
    return found


# ─────────────────────────────────────────────────────────────────
# 5. REDUNDANT CONTEXT DETECTION
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
                redundant.append({"s1": sentences[i][:80], "s2": sentences[j][:80], "similarity": round(sim, 2)})

    removable_tokens = sum(token_count(p["s2"]) for p in redundant)
    return {"redundant_pairs": redundant, "estimated_removable_tokens": removable_tokens}


# ─────────────────────────────────────────────────────────────────
# 6. COMPRESSION + GARDE SÉMANTIQUE
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
    orig_doc = nlp_model(original)
    comp_doc = nlp_model(compressed)

    orig_nouns = {t.lemma_.lower() for t in orig_doc if t.pos_ == "NOUN" and not t.is_stop}
    if not orig_nouns:
        return compressed

    comp_nouns = {t.lemma_.lower() for t in comp_doc if t.pos_ == "NOUN" and not t.is_stop}
    ratio = len(comp_nouns & orig_nouns) / len(orig_nouns)

    if ratio < threshold:
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
    result = re.sub(r"[!?.]{3,}", lambda m: m.group(0)[0], result)
    result = re.sub(r"(\n\s*){3,}", "\n\n", result)
    result = re.sub(r"\s{2,}", " ", result).strip()
    result = _preserve_unique_nouns(prompt, result)
    return result


# ─────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────
@dataclass
class EconomyReport:
    original_prompt: str
    compressed_prompt: str
    original_tokens: int
    compressed_tokens: int
    token_reduction: int
    reduction_pct: float
    fillers_removed: list[dict]
    repetitions: dict
    language_analysis: dict
    costly_patterns: list[dict]
    redundant_sentences: dict


def analyze(prompt: str) -> EconomyReport:
    """
    Full token economy analysis. Called by app.py.
    Returns EconomyReport with all findings and the compressed prompt.
    """
    compressed = compress(prompt)
    _, fillers = remove_fillers(prompt)
    repetitions = detect_repetitions(prompt)
    language_analysis = detect_languages(prompt)
    costly = detect_costly_patterns(prompt)
    redundant = detect_redundant_sentences(prompt)

    orig_tok = token_count(prompt)
    comp_tok = token_count(compressed)

    return EconomyReport(
        original_prompt=prompt,
        compressed_prompt=compressed,
        original_tokens=orig_tok,
        compressed_tokens=comp_tok,
        token_reduction=orig_tok - comp_tok,
        reduction_pct=round((1 - comp_tok / max(orig_tok, 1)) * 100, 1),
        fillers_removed=fillers,
        repetitions=repetitions,
        language_analysis=language_analysis,
        costly_patterns=costly,
        redundant_sentences=redundant,
    )
