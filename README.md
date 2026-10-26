
# CineMatch — Movie Recommendation System

CineMatch is a Streamlit application powered by a trained SVD collaborative-filtering model.

## Features

- Personalized movie recommendations
- Predicted ratings for selected movies
- Genre-based movie similarity using Jaccard overlap
- Model-comparison metrics
- Ranking evaluation metrics
- Dataset and model metadata

## Technology Stack

- Python 3.11
- Streamlit
- Pandas and NumPy
- Scikit-Surprise
- Joblib
- Plotly

## Project Setup

Activate the virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Start the application:

```powershell
python -m streamlit run app.py
```

Open the local URL printed in the terminal, usually http://localhost:8501.

## Model Artifacts

Place these six files inside the `models/` directory:

- `best_model.joblib`
- `movies_clean.pkl`
- `ratings_model.pkl`
- `model_comparison.csv`
- `ranking_metrics.csv`
- `metadata.json`

The app loads the saved model and does not retrain it.

## Methodology Notes

The main recommendation engine uses SVD collaborative filtering. The similar-movies feature uses Jaccard similarity over genre labels and is a separate, simple content-based feature.

Ranking metrics depend on the candidate set and relevance threshold used during evaluation.

The available ratings file contained fewer records than the canonical MovieLens 20M dataset. Use the actual data counts in your project report.
