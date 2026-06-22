#  Titanic Survival Clustering

> Unsupervised Machine Learning project that segments Titanic passengers into meaningful survival profiles using **K-Means** and **DBSCAN** clustering — without relying on labels during training.

---

## Overview

Most Titanic analyses focus on **supervised classification** (predicting who survives). This project takes a different angle: using **unsupervised clustering** to discover hidden passenger profiles purely from the data structure. The goal is to answer:

> *"Can we identify natural passenger groups that align with survival patterns — without ever telling the model who survived?"*

---

##  ML Pipeline
Raw Data → EDA → Feature Engineering → Normalization → Clustering → Validation → Visualization

| Stage | Details |
|---|---|
| **Data Source** | Titanic dataset via `fetch_openml` (OpenML ID: 40945) |
| **EDA** | Survival rates by sex, class, age distribution, correlation heatmap |
| **Feature Engineering** | `FamilyNb` (SibSp + Parch), `Alone` flag, One-Hot Encoding |
| **Preprocessing** | MinMaxScaler — full normalization to [0,1] |
| **Algorithm 1** | K-Means (k=3, k-means++ initialization) |
| **Algorithm 2** | DBSCAN (density-based, automatic outlier detection) |
| **Validation** | Elbow Method, Silhouette Score |
| **Visualization** | PCA 2D projection, cluster heatmaps, scatter plots |

---

##  Key Findings

### K-Means — 3 Passenger Profiles

| Cluster | Profile | Survival Rate |
|---|---|---|
| 0 | Female, 1st/2nd class, traveling with family |  High |
| 1 | Male, 3rd class, young, alone |  Low |
| 2 | Mixed age, intermediate class |  Medium |

### DBSCAN — Outlier Detection
- Detected **~27 statistical outliers** — passengers with atypical combinations of age, class, and family size that don't fit any natural cluster
- Confirms that K-Means alone would force these anomalies into incorrect groups

### Validation
- **Silhouette Score** confirms meaningful cluster separation for K=3
- **Elbow Method** validates K=3 as the optimal number of clusters

---

##  Project Structure
titanic-survival-clustering/

│

├── titanic_dataset.ipynb    # Full ML pipeline notebook

├── titanic_train.csv        # Raw dataset (download from Kaggle)

└── README.md

---

##  Setup & Usage

### 1. Clone the repository
```bash
git clone https://github.com/soltanioumayma/titanic-survival-clustering.git
cd titanic-survival-clustering
```

### 2. Install dependencies
```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

### 3. Get the dataset
Download `train.csv` from [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic/data), rename it to `titanic_train.csv` and place it in the project root.

### 4. Run the notebook
```bash
jupyter notebook titanic_dataset.ipynb
```

---

##  Visualizations Included

-  Survival distribution by sex and passenger class
-  Age distribution by survival outcome
-  Pearson correlation heatmap
-  Cluster center signature heatmap
-  Elbow curve (WCSS vs K)
-  PCA 2D scatter — K-Means vs DBSCAN side by side

---

##  Tech Stack

| Library | Purpose |
|---|---|
| `pandas` | Data manipulation |
| `numpy` | Numerical computing |
| `scikit-learn` | ML algorithms, PCA, scaling |
| `matplotlib` | Plotting |
| `seaborn` | Statistical visualizations |

---

##  Author

**Oumayma Soltani**
- GitHub: [soltanioumayma](https://github.com/soltanioumayma)
- LinkedIn: [soltanioumayma](https://linkedin.com/in/soltanioumayma)

---

## 📄 License

This project is licensed under the MIT License — feel free to use, modify, and share.
