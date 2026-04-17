# EcoPrompt AI (MVP)

Prototype Streamlit app that optimizes prompts to cut token waste, classify the task, and estimate energy/CO₂ savings. Everything runs locally with lightweight scikit-learn models and public datasets.

## Quick start
1. Install deps: `pip install -r requirements.txt`
2. Download spaCy English model: `python -m spacy download en_core_web_sm`
3. Train the classifiers (downloads datasets automatically): `python train.py`
4. Run the demo: `streamlit run app.py`

## What it does
- Clarity gate detects ambiguous prompts and offers checkbox clarifications.
- Token economy engine removes fillers, flags repetition/multi-language/costly patterns, finds redundant sentences, and compresses the prompt while keeping ≥85% of unique nouns.
- Task classifier routes to a recommended model tier and provides a lightweight complexity score.
- Energy estimator computes Wh and CO₂ savings versus a heavy-model baseline.

## Project layout
- `app.py` — Streamlit UI
- `train.py` — trains `models/ambiguity_clf.joblib` and `models/task_clf.joblib`
- `clarity_gate.py` — ambiguity detection + clarification prompts
- `token_economist.py` — token analysis & compression
- `task_classifier.py` — task prediction
- `energy_estimator.py` — energy/CO₂ calculations
- `models/` — saved joblib models (created by `train.py`)
- `data/` — datasets downloaded via HuggingFace

## Notes
- No external LLMs or GPUs required.
- Datasets are ~150 MB total. Training takes a few minutes on CPU.
- Energy figures use Luccioni et al. 2023 averages and are approximate.
