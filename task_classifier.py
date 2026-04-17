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
    scores = model.decision_function([prompt])[0]
    confidence = round(float(max(scores) / (sum(abs(s) for s in scores) + 1e-9)), 2)
    return {"task": label, "confidence": confidence}
