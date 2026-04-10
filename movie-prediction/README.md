# Movie Box Office Prediction

A machine learning pipeline that predicts whether a movie will be **profitable** based on budget, genre, popularity, and audience ratings — using Decision Tree and Logistic Regression classifiers.

![Prediction Preview](movie_prediction.png)

## Overview

This project applies binary classification to the film industry. Given pre-release and metadata features, the model predicts whether a movie's box office revenue will exceed 1.5× its production budget.

**Tech Stack:** Python · scikit-learn · pandas · matplotlib · seaborn

## Features

- Loads the **TMDB 5000 Movies** dataset or falls back to **synthetic data**
- Feature engineering: JSON genre parsing, release season extraction, one-hot encoding
- Two models compared: Decision Tree (interpretable) vs Logistic Regression (generalised)
- Profitability threshold: Revenue > 1.5 × Budget
- 6-panel visualisation dashboard saved as PNG

## Project Structure

```
movie-prediction/
├── movie_prediction.py      # Main script: data loading, models, charts
├── movie_prediction.png     # Output chart (generated after running)
└── README.md
```

## Getting Started

### Option A — Real Data (Recommended)

1. Download from Kaggle:  
   [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)

2. Place `tmdb_5000_movies.csv` in this folder.

3. Run:
   ```bash
   pip install pandas scikit-learn matplotlib seaborn
   python movie_prediction.py
   ```

### Option B — Synthetic Data (No download needed)

```bash
pip install pandas scikit-learn matplotlib seaborn
python movie_prediction.py
```

The script generates 300 synthetic movies with realistic distributions and noise.

## Features Used

| Feature | Description |
|---|---|
| `budget_M` | Production budget (millions USD) |
| `runtime_min` | Film runtime in minutes |
| `popularity` | TMDB popularity score |
| `vote_avg` | Audience vote average (1–10) |
| `vote_count` | Number of votes |
| `sequel` | Is the film a sequel (0/1) |
| `star_power` | Lead actor popularity score (1–10) |
| `genre_*` | One-hot encoded primary genre |
| `season_*` | Release season (Spring/Summer/Fall/Winter) |

## Model Results (Synthetic Data)

| Model | Accuracy |
|---|---|
| Decision Tree (depth=5) | ~57% |
| Logistic Regression | ~73% |

Results on real TMDB data will vary based on label balance.

## Visualisation Panels

- **Budget vs Revenue** — scatter with break-even line at 1.5×
- **Feature Importances** — top 8 decision tree features
- **Confusion Matrix** — predicted vs actual for Decision Tree
- **Model Accuracy** — side-by-side bar comparison
- **Popularity vs Rating** — coloured by profitability
- **Profit Rate by Genre** — which genres tend to be most profitable

## Author

Caiyu Qu · [github.com/amberquu](https://github.com/amberquu)
