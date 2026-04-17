from pathlib import Path

import joblib
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


def _already_exists(path: Path, label: str) -> bool:
    if path.exists():
        print(f"✓ {label} already exists — skipping.")
        return True
    return False


def maybe_train_ambiguity() -> None:
    path = MODEL_DIR / "ambiguity_clf.joblib"
    if _already_exists(path, "ambiguity_clf.joblib"):
        return

    ambig = load_dataset("ambig_qa", "light", split="train")
    ambig_prompts = [row["question"] for row in ambig]

    pairs = load_dataset("Jayveersinh-Raj/bad-improved-prompt-pairs", split="train")
    clear_prompts = [
        row["improved_prompt"]
        for row in pairs
        if row.get("improved_prompt")
    ][: len(ambig_prompts)]

    texts = ambig_prompts + clear_prompts
    labels = [1] * len(ambig_prompts) + [0] * len(clear_prompts)

    ambiguity_pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=30000,
                    ngram_range=(1, 3),
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=1.0,
                    max_iter=1000,
                    class_weight="balanced",
                ),
            ),
        ]
    )
    ambiguity_pipeline.fit(texts, labels)
    joblib.dump(ambiguity_pipeline, path)
    print("✓ Ambiguity classifier saved")


TASK_KEYWORDS = {
    "code": ["function", "bug", "script", "code", "debug", "implement", "class", "api"],
    "writing": ["write", "essay", "email", "letter", "blog", "article", "story"],
    "reasoning": ["analyze", "compare", "pros", "cons", "argue", "evaluate", "strategy"],
    "summarization": ["summarize", "summary", "tldr", "brief", "shorten", "condense"],
    "qa": ["what is", "who is", "define", "explain", "when was", "how does"],
    "data": ["dataset", "csv", "dataframe", "chart", "plot", "statistics", "pandas"],
    "translation": ["translate", "in french", "in arabic", "in spanish", "en français"],
    "creative": ["poem", "story", "imagine", "create", "invent", "fictional"],
}


def auto_label(text: str) -> str:
    t = text.lower()
    for task, keywords in TASK_KEYWORDS.items():
        if any(keyword in t for keyword in keywords):
            return task
    return "general"


def maybe_train_task() -> None:
    path = MODEL_DIR / "task_clf.joblib"
    if _already_exists(path, "task_clf.joblib"):
        return

    wizard = load_dataset("WizardLM/WizardLM_evol_instruct_70k", split="train")
    wiz_texts = [row["instruction"] for row in wizard]
    wiz_labels = [auto_label(text) for text in wiz_texts]

    task_pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=50000, ngram_range=(1, 2))),
            ("clf", LinearSVC(C=1.0, max_iter=2000)),
        ]
    )
    task_pipeline.fit(wiz_texts, wiz_labels)
    joblib.dump(task_pipeline, path)
    print("✓ Task classifier saved")


if __name__ == "__main__":
    maybe_train_ambiguity()
    maybe_train_task()
    print("Done. Run: streamlit run app.py")
