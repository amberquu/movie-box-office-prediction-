"""
movie_prediction.py — Movie Box Office Prediction
Author: Caiyu Qu | github.com/amberquu

━━ Real Dataset (recommended) ━━━━━━━━━━━━━━━━━━━━━━━━
Download from Kaggle:
  https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata
Place "tmdb_5000_movies.csv" in the same folder, then run:
  python movie_prediction.py

━━ No dataset? ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The script auto-generates synthetic data as a fallback.

Run: pip install pandas scikit-learn matplotlib seaborn
     python movie_prediction.py
"""
import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

np.random.seed(42)

# ══════════════════════════════════════════════════════
#  1.  LOAD DATA
# ══════════════════════════════════════════════════════
TMDB_FILE = "tmdb_5000_movies.csv"

def load_tmdb(path):
    """Load and clean the TMDB 5000 dataset."""
    raw = pd.read_csv(path)

    df = pd.DataFrame()
    df["title"]      = raw["title"]
    df["budget_M"]   = raw["budget"] / 1e6
    df["revenue_M"]  = raw["revenue"] / 1e6
    df["runtime_min"]= raw["runtime"].fillna(raw["runtime"].median())
    df["popularity"] = raw["popularity"]
    df["vote_avg"]   = raw["vote_average"]
    df["vote_count"] = raw["vote_count"]

    # Parse genres (JSON string) → primary genre
    def primary_genre(g):
        try:
            gs = json.loads(g.replace("'",'"'))
            return gs[0]["name"] if gs else "Unknown"
        except:
            return "Unknown"
    df["genre"] = raw["genres"].apply(primary_genre)

    # Release season from release_date
    def season(d):
        try:
            m = pd.to_datetime(d).month
            return ["Winter","Spring","Spring","Summer","Summer","Summer",
                    "Fall","Fall","Fall","Winter","Winter","Winter"][m-1]
        except:
            return "Unknown"
    df["season"] = raw["release_date"].apply(season)

    # Drop rows with no budget or revenue (unreported)
    df = df[(df["budget_M"] > 0.5) & (df["revenue_M"] > 0.5)].reset_index(drop=True)

    # Label: profitable = revenue > 1.5× budget
    df["profitable"] = (df["revenue_M"] > df["budget_M"] * 1.5).astype(int)
    df["star_power"] = 5   # not in dataset; set neutral
    df["sequel"]     = 0

    print(f"[TMDB] Loaded {len(df)} movies | "
          f"Profitable: {df['profitable'].mean()*100:.1f}%")
    return df


def generate_synthetic():
    """Fallback: synthetic dataset when no CSV is present."""
    print("[Synthetic] TMDB CSV not found — using generated data.")
    print("  → Download real data: https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata\n")
    N = 300
    genres  = ["Action","Comedy","Drama","Horror","Sci-Fi","Romance","Animation"]
    seasons = ["Spring","Summer","Fall","Winter"]
    df = pd.DataFrame({
        "title":       [f"Movie_{i:03d}" for i in range(N)],
        "budget_M":    np.round(np.random.lognormal(3.5, 0.9, N), 1),
        "runtime_min": np.random.randint(80, 180, N),
        "popularity":  np.round(np.random.uniform(10, 100, N), 1),
        "vote_avg":    np.round(np.random.normal(6.2, 1.2, N).clip(1,10), 1),
        "vote_count":  np.random.randint(100, 20000, N),
        "genre":       np.random.choice(genres, N),
        "season":      np.random.choice(seasons, N),
        "sequel":      np.random.choice([0,1], N, p=[0.65,0.35]),
        "star_power":  np.random.randint(1, 11, N),
    })
    revenue = (df["popularity"]*0.5 + df["vote_avg"]*5 +
               df["star_power"]*3 + df["sequel"]*15 +
               np.random.normal(0, 80, N)).clip(1)
    df["revenue_M"]  = np.round(revenue, 1)
    df["profitable"] = (df["revenue_M"] > df["budget_M"]).astype(int)
    print(f"[Synthetic] {len(df)} movies | Profitable: {df['profitable'].mean()*100:.1f}%\n")
    return df


df = load_tmdb(TMDB_FILE) if os.path.exists(TMDB_FILE) else generate_synthetic()

# ══════════════════════════════════════════════════════
#  2.  FEATURE ENGINEERING
# ══════════════════════════════════════════════════════
df_enc = pd.get_dummies(df, columns=["genre","season"], drop_first=True)
base_features = ["budget_M","runtime_min","popularity","vote_avg","vote_count","sequel","star_power"]
extra = [c for c in df_enc.columns if c.startswith(("genre_","season_"))]
features = [f for f in base_features + extra if f in df_enc.columns]

X = df_enc[features].fillna(0)
y = df_enc["profitable"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ══════════════════════════════════════════════════════
#  3.  MODELS
# ══════════════════════════════════════════════════════
dt = DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, random_state=42)
dt.fit(X_train, y_train)
dt_preds = dt.predict(X_test)
dt_acc   = accuracy_score(y_test, dt_preds)

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_s, y_train)
lr_preds = lr.predict(X_test_s)
lr_acc   = accuracy_score(y_test, lr_preds)

print(f"Decision Tree      accuracy: {dt_acc*100:.1f}%")
print(f"Logistic Regression accuracy: {lr_acc*100:.1f}%\n")
print("── Decision Tree Report ──")
print(classification_report(y_test, dt_preds, target_names=["Flop","Hit"]))

# ══════════════════════════════════════════════════════
#  4.  VISUALISATIONS
# ══════════════════════════════════════════════════════
plt.style.use("seaborn-v0_8-whitegrid")
fig, axes = plt.subplots(2, 3, figsize=(17, 10))
data_label = "TMDB" if os.path.exists(TMDB_FILE) else "Synthetic"
fig.suptitle(f"Movie Box Office Prediction ({data_label} Data)", fontsize=15, fontweight="bold")

from matplotlib.patches import Patch
colors = df["profitable"].map({1:"#5B8FF9", 0:"#E8684A"})
legend_patches = [Patch(color="#5B8FF9",label="Profitable"), Patch(color="#E8684A",label="Flop")]

# (A) Budget vs Revenue
ax = axes[0,0]
ax.scatter(df["budget_M"].clip(upper=400), df["revenue_M"].clip(upper=800),
           c=colors, alpha=0.45, s=20)
lim = min(400, df["budget_M"].quantile(.95))
ax.plot([0,lim],[0,lim*1.5],"k--",lw=1,label="Break-even (1.5×)")
ax.set_xlabel("Budget ($M)"); ax.set_ylabel("Revenue ($M)")
ax.set_title("Budget vs Revenue")
ax.legend(handles=legend_patches+[plt.Line2D([0],[0],color="k",linestyle="--",label="Break-even")], fontsize=8)

# (B) Feature importance
ax = axes[0,1]
imp = pd.Series(dt.feature_importances_, index=features).nlargest(8)
ax.barh(imp.index, imp.values, color="#5AD8A6")
ax.set_title("Top 8 Feature Importances (DT)")
ax.set_xlabel("Importance"); ax.invert_yaxis()

# (C) Confusion matrix
ax = axes[0,2]
cm = confusion_matrix(y_test, dt_preds)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Flop","Hit"], yticklabels=["Flop","Hit"])
ax.set_title("Confusion Matrix (Decision Tree)")
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")

# (D) Model accuracy comparison
ax = axes[1,0]
bars = ax.bar(["Decision Tree","Logistic Regression"], [dt_acc*100, lr_acc*100],
              color=["#5B8FF9","#F6BD16"], width=0.4)
ax.set_ylim(0,105); ax.set_ylabel("Accuracy (%)")
ax.set_title("Model Accuracy Comparison")
for bar, acc in zip(bars, [dt_acc*100, lr_acc*100]):
    ax.text(bar.get_x()+bar.get_width()/2, acc+1, f"{acc:.1f}%", ha="center", fontweight="bold")

# (E) Popularity vs Vote Average
ax = axes[1,1]
ax.scatter(df["popularity"].clip(upper=200), df["vote_avg"], c=colors, alpha=0.45, s=20)
ax.set_xlabel("Popularity"); ax.set_ylabel("Vote Average")
ax.set_title("Popularity vs Rating")
ax.legend(handles=legend_patches, fontsize=8)

# (F) Profit rate by genre
ax = axes[1,2]
top_genres = df.groupby("genre")["profitable"].mean().nlargest(7)
ax.bar(top_genres.index, top_genres.values*100, color="#9270CA")
ax.set_title("Profit Rate by Genre"); ax.set_ylabel("Profit Rate (%)")
ax.tick_params(axis="x", rotation=30)
for i,v in enumerate(top_genres.values*100):
    ax.text(i, v+0.5, f"{v:.0f}%", ha="center", fontsize=8)

plt.tight_layout()
plt.savefig("movie_prediction.png", dpi=150, bbox_inches="tight")
print("Saved: movie_prediction.png")
plt.show()
