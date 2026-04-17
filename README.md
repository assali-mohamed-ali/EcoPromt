# EcoPrompt AI MVP

Prototype Streamlit pour démontrer l'optimisation de prompts et l'impact énergétique estimé.

## Stack
- streamlit
- scikit-learn
- spacy
- tiktoken
- pandas / numpy
- datasets

## Setup
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python train.py
streamlit run app.py
```

## Modules
- `train.py`: entraîne `models/ambiguity_clf.joblib` et `models/task_clf.joblib`
- `clarity_gate.py`: détection d'ambiguïté + questions de clarification
- `token_economist.py`: détection de gaspillage de tokens + compression
- `task_classifier.py`: classification du type de tâche
- `energy_estimator.py`: estimation énergie / CO₂
- `app.py`: UI Streamlit
