from __future__ import annotations

from dataclasses import dataclass, field

import joblib

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
        "options": [
            "Write new code",
            "Debug / fix a bug",
            "Explain existing code",
            "Refactor / optimize",
            "Add tests",
            "Convert to another language",
        ],
    },
    "writing": {
        "id": "writing_goal",
        "text": "What kind of writing?",
        "options": [
            "Professional / formal",
            "Casual / conversational",
            "Technical doc",
            "Creative / storytelling",
            "Email draft",
        ],
    },
    "general": {
        "id": "general_goal",
        "text": "What is the main goal of your request?",
        "options": [
            "Get information",
            "Create something",
            "Analyze or compare",
            "Solve a technical problem",
            "Summarize content",
        ],
    },
}

LENGTH_Q = {
    "id": "output_length",
    "text": "How long should the response be?",
    "options": [
        "One-liner",
        "Short paragraph",
        "Detailed explanation",
        "Comprehensive with examples",
    ],
}


def detect_domain(prompt: str) -> str:
    p = prompt.lower()
    if any(k in p for k in ["code", "function", "bug", "script", "python", "javascript"]):
        return "code"
    if any(k in p for k in ["write", "essay", "email", "letter", "story", "blog"]):
        return "writing"
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
            lines.append(f"[Context — {qid.replace('_', ' ')}: {', '.join(selections)}]")
    return "\n".join(lines)
