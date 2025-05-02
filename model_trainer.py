import os
import logging
import pickle
import joblib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    RocCurveDisplay,
)
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# ───────────────────────────────────────────────────────────────────────────────
# Configuration & Logging
# ───────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
RANDOM_STATE = 42
DATA_PATH    = "./datasets/dataset.csv"
MODEL_DIR    = "model"
VIS_DIR      = "visualizations"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)

# ───────────────────────────────────────────────────────────────────────────────
# 1. Load & Inspect Data
# ───────────────────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH, low_memory=False)
logging.info(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")

# ───────────────────────────────────────────────────────────────────────────────
# 2. Prepare Features & Target
# ───────────────────────────────────────────────────────────────────────────────
X = df.drop(columns=["ID", "md5", "legitimate"])
y = df["legitimate"]

# ───────────────────────────────────────────────────────────────────────────────
# 3. Feature Selection via ExtraTrees + SelectFromModel
# ───────────────────────────────────────────────────────────────────────────────
etc = ExtraTreesClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
etc.fit(X, y)
selector = SelectFromModel(etc, prefit=True, threshold="median")
X_sel = selector.transform(X)
feat_names = X.columns[selector.get_support()].tolist()
logging.info(f"Selected {len(feat_names)} features via ExtraTrees")

# ───────────────────────────────────────────────────────────────────────────────
# 4. Train/Test Split (Stratified)
# ───────────────────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_sel, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)
logging.info(f"Split data: {X_train.shape[0]} train, {X_test.shape[0]} test samples")

# ───────────────────────────────────────────────────────────────────────────────
# 5. Hyperparameter Tuning for RandomForest
# ───────────────────────────────────────────────────────────────────────────────
param_grid = {
    "n_estimators": [10, 33, 50],
    "max_depth": [None, 10, 20]
}
rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
grid = GridSearchCV(rf, param_grid, cv=cv, scoring="f1", n_jobs=-1)
grid.fit(X_train, y_train)

best_rf = grid.best_estimator_
logging.info(f"Best RF params: {grid.best_params_}")

# ───────────────────────────────────────────────────────────────────────────────
# 6. Evaluation on Test Set
# ───────────────────────────────────────────────────────────────────────────────
y_pred  = best_rf.predict(X_test)
y_proba = best_rf.predict_proba(X_test)[:, 1]

acc     = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)
cm      = confusion_matrix(y_test, y_pred)

logging.info(f"Test Accuracy: {acc:.4f}")
logging.info(f"Test ROC-AUC:  {roc_auc:.4f}")
logging.info("\n" + classification_report(y_test, y_pred, target_names=["Malware","Legit"]))

# ───────────────────────────────────────────────────────────────────────────────
# 7. Save Model & Selected Features
# ───────────────────────────────────────────────────────────────────────────────
model_path    = os.path.join(MODEL_DIR, "model.pkl")
features_path = os.path.join(MODEL_DIR, "features.pkl")

joblib.dump(best_rf, model_path)
with open(features_path, "wb") as f:
    pickle.dump(feat_names, f)

logging.info(f"Saved model to: {model_path}")
logging.info(f"Saved selected features to: {features_path}")

# ───────────────────────────────────────────────────────────────────────────────
# 8. Visualizations
# ───────────────────────────────────────────────────────────────────────────────
sns.set_context("paper")
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 200

# 8.1 Class Distribution Barplot
df_counts = pd.DataFrame({
    "label": ["Malware","Legit"],
    "count": y.value_counts().sort_index().values
})
plt.figure(figsize=(5,4))
sns.barplot(data=df_counts, x="label", y="count", hue="label", dodge=False,
            palette=["#e74c3c","#2ecc71"], legend=False)
plt.title("Class Distribution")
plt.ylabel("Sample Count")
plt.xlabel("")
plt.tight_layout()
plt.savefig(f"{VIS_DIR}/class_distribution_bar.png")
plt.close()

# 8.2 Class Distribution Pie Chart
plt.figure(figsize=(5,5))
df_counts.set_index("label")["count"].plot.pie(
    autopct="%1.1f%%", colors=["#e74c3c","#2ecc71"]
)
plt.title("Class Proportion")
plt.ylabel("")
plt.tight_layout()
plt.savefig(f"{VIS_DIR}/class_distribution_pie.png")
plt.close()

# 8.3 Confusion Matrix Heatmap
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Malware","Legit"], yticklabels=["Malware","Legit"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(f"{VIS_DIR}/confusion_matrix.png")
plt.close()

# 8.4 ROC Curve
plt.figure(figsize=(6,5))
RocCurveDisplay.from_predictions(y_test, y_proba, name="RandomForest")
plt.title("ROC Curve")
plt.tight_layout()
plt.savefig(f"{VIS_DIR}/roc_curve.png")
plt.close()

# 8.5 Feature Importance Barplot (Top 15)
importances = pd.Series(best_rf.feature_importances_, index=feat_names).sort_values(ascending=False).head(15)
imp_df = importances.reset_index()
imp_df.columns = ["feature","importance"]

plt.figure(figsize=(8,6))
sns.barplot(data=imp_df, x="importance", y="feature", hue="feature", dodge=False,
            palette="viridis", legend=False)
plt.title("Top 15 Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig(f"{VIS_DIR}/feature_importance.png")
plt.close()

# 8.6 Correlation Among Top 10 Features by Correlation with Label
# Build DataFrame of selected features + label
df_vis = pd.DataFrame(X_sel, index=df.index, columns=feat_names)
df_vis["legitimate"] = y.values

corr_with_label = df_vis.corrwith(df_vis["legitimate"]).abs().sort_values(ascending=False)
top10 = corr_with_label.drop("legitimate").head(10).index.tolist()

plt.figure(figsize=(10,8))
sns.heatmap(
    df_vis[top10].corr().abs(),
    annot=True, fmt=".2f", cmap="vlag", cbar_kws={"label":"Abs Corr"}
)
plt.title("Inter-Feature Correlation (Top 10 by Corr with Label)")
plt.tight_layout()
plt.savefig(f"{VIS_DIR}/selected_features_correlation.png")
plt.close()

# 8.7 PCA 2D Projection (sampled)
sample_idx = np.random.choice(X_sel.shape[0], size=2000, replace=False)
pca_proj = PCA(n_components=2, random_state=RANDOM_STATE)\
           .fit_transform(StandardScaler().fit_transform(X_sel[sample_idx]))
plt.figure(figsize=(6,5))
sns.scatterplot(
    x=pca_proj[:,0], y=pca_proj[:,1], hue=y.iloc[sample_idx],
    palette=["#e74c3c","#2ecc71"], alpha=0.6, s=20
)
plt.title("PCA 2D Projection")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.savefig(f"{VIS_DIR}/pca_projection.png")
plt.close()

# 8.8 t-SNE 2D Projection (sampled)
tsne_proj = TSNE(n_components=2, perplexity=30, n_iter=500, random_state=RANDOM_STATE)\
            .fit_transform(StandardScaler().fit_transform(X_sel[sample_idx]))
plt.figure(figsize=(6,5))
sns.scatterplot(
    x=tsne_proj[:,0], y=tsne_proj[:,1], hue=y.iloc[sample_idx],
    palette=["#e74c3c","#2ecc71"], alpha=0.6, s=20
)
plt.title("t-SNE 2D Projection")
plt.xlabel("Dim1")
plt.ylabel("Dim2")
plt.tight_layout()
plt.savefig(f"{VIS_DIR}/tsne_projection.png")
plt.close()

logging.info(f"All visualizations saved to '{VIS_DIR}/'")
