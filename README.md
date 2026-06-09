# FIFA-WORLD-CUP-MATCHES-PREDICTION
# 🏆 FIFA 2026 World Cup - AI Match Predictor

An enterprise-grade machine learning pipeline designed to predict international football match outcomes, calculate Expected Goals (xG), and simulate the upcoming FIFA World Cup 2026 group stages. 

Unlike standard prediction scripts, this project utilizes a **chronological, zero-leakage feature engineering architecture**, a dynamic custom Elo rating system, and SHAP-based model explainability.

## ✨ Key Features

* **Zero-Leakage Feature Engineering:** Simulates chronological match history, ensuring that predictions only use data (rolling form, goals scored/conceded, H2H stats) available *prior* to kickoff.
* **Dynamic Elo Rating Engine:** Custom-built ranking system that adjusts team strength based on match importance (e.g., World Cup vs. Friendlies) and margin of victory.
* **Parametric xG Model:** Blends historical attack/defense intensities with Elo differentials to project Expected Goals (xG) for every matchup.
* **Algorithm Benchmarking:** Automated cross-validation testing across Logistic Regression, Random Forest, and XGBoost classifiers.
* **SHAP Explainability:** Uses Game Theory (SHapley Additive exPlanations) to break down "black-box" predictions into plain-English driving factors.
* **Interactive UI Dashboard:** A front-end interface built directly into Jupyter Notebooks (`ipywidgets`) allowing users to simulate matches on the fly.

## 🛠️ Tech Stack

* **Language:** Python 3.9+
* **Machine Learning:** XGBoost, Scikit-Learn
* **Data Processing:** Pandas, NumPy
* **Explainability & Visualization:** SHAP, Matplotlib, Seaborn
* **Environment:** Jupyter Notebooks, IPyWidgets

## 📂 Project Structure

```text
fifa-ml-project/
│
├── data/                                 # Raw data directory
│   └── international_results.csv         # Historical dataset (auto-downloaded)
│
├── models/                               # Serialized model artifacts
│   └── fifa_xgboost.joblib               # Trained XGBoost engine
│
├── notebooks/                            # Core execution & UI layer
│   ├── 04_Model_Training_and_Simulation.ipynb  # Pipeline backend
│   └── 05_Final_Predictor.ipynb                # Interactive dashboard
│
├── visuals/                              # Generated plots (SHAP, feature importance)
│
└── README.md                             # Project documentation
