import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier

# 1. Load full dataset
df = pd.read_csv("datasets/dataset.csv")

# 2. Prepare output folder
output_dir = "visualizations_full"
os.makedirs(output_dir, exist_ok=True)

# 3. Drop identifiers
df_model = df.drop(columns=["ID", "md5"])

# 4. Split features & target
X_full = df_model.drop(columns=["legitimate"])
y_full = df_model["legitimate"]

# 5. Sample for heavy computations
SAMPLE_SIZE = 10000
df_sample = df_model.sample(n=SAMPLE_SIZE, random_state=42)
X = df_sample.drop(columns=["legitimate"])
y = df_sample["legitimate"]

# 6. Standardize sample for PCA/TSNE
X_scaled = StandardScaler().fit_transform(X)

# 7. Plot style
sns.set_context("paper")
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 300

# ── 1) Class distribution (bar) ──────────────────────────────────────────────
counts = y_full.value_counts().sort_index()
plt.figure(figsize=(5,4))
plt.bar([0,1], counts.values, color=["#e74c3c","#2ecc71"])
plt.xticks([0,1], ["Malware (0)","Legitimate (1)"])
plt.title("Class Distribution")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(f"{output_dir}/01_class_distribution.png")
plt.close()

# ── 2) Class proportion (pie) ───────────────────────────────────────────────
plt.figure(figsize=(5,5))
counts.plot.pie(autopct="%1.1f%%", labels=["Malware","Legit"], colors=["#e74c3c","#2ecc71"])
plt.title("Class Proportion")
plt.ylabel("")
plt.tight_layout()
plt.savefig(f"{output_dir}/02_class_proportion.png")
plt.close()

# ── 3) Correlation heatmap (top 20) ────────────────────────────────────────
corr = df_sample.corr()
top20 = corr["legitimate"].abs().sort_values(ascending=False).head(21).index
plt.figure(figsize=(12,10))
sns.heatmap(df_sample[top20].corr(), annot=True, fmt=".2f", cbar_kws={'label':'Corr'}, cmap="vlag")
plt.title("Top 20 Feature Correlations with Label")
plt.tight_layout()
plt.savefig(f"{output_dir}/03_top20_corr_heatmap.png")
plt.close()

# ── 4) KDE plots for top-4 correlated features ──────────────────────────────
top4 = corr["legitimate"].abs().drop("legitimate").sort_values(ascending=False).head(4).index
for feat in top4:
    plt.figure(figsize=(6,4))
    sns.kdeplot(data=df_sample, x=feat, hue="legitimate", fill=True, common_norm=False)
    plt.title(f"KDE of {feat}")
    plt.xlabel(feat)
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/04_kde_{feat}.png")
    plt.close()

# ── 5) Violin plots for top-4 variance features ─────────────────────────────
top_var = df_sample.var().sort_values(ascending=False).drop("legitimate").head(4).index
for feat in top_var:
    plt.figure(figsize=(6,4))
    sns.violinplot(x="legitimate", y=feat, data=df_sample, palette=["#e74c3c","#2ecc71"])
    plt.title(f"Violin Plot: {feat}")
    plt.xlabel("Label")
    plt.ylabel(feat)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/05_violin_{feat}.png")
    plt.close()

# ── 6) Histograms for those same variance features ─────────────────────────
for feat in top_var:
    plt.figure(figsize=(6,4))
    sns.histplot(data=df_sample, x=feat, hue="legitimate", kde=True, element="step", palette=["#e74c3c","#2ecc71"])
    plt.title(f"Histogram: {feat}")
    plt.xlabel(feat)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/06_hist_{feat}.png")
    plt.close()

# ── 7) Scatter: Resource vs Section Entropy ────────────────────────────────
plt.figure(figsize=(6,5))
sns.scatterplot(data=df_sample, x="ResourcesMeanEntropy", y="SectionsMeanEntropy",
                hue="legitimate", alpha=0.5, palette=["#e74c3c","#2ecc71"])
plt.title("Resource vs Section Entropy")
plt.xlabel("ResourcesMeanEntropy")
plt.ylabel("SectionsMeanEntropy")
plt.tight_layout()
plt.savefig(f"{output_dir}/07_scatter_entropy.png")
plt.close()

# ── 8) PCA projection (2D) ────────────────────────────────────────────────
proj = PCA(n_components=2).fit_transform(X_scaled)
plt.figure(figsize=(6,5))
sns.scatterplot(x=proj[:,0], y=proj[:,1], hue=y, alpha=0.5, palette=["#e74c3c","#2ecc71"])
plt.title("PCA: 2D Projection")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.savefig(f"{output_dir}/08_pca_projection.png")
plt.close()

# ── 9) t-SNE projection (2D) ──────────────────────────────────────────────
sample_idx = np.random.choice(X_scaled.shape[0], size=5000, replace=False)
tsne_proj = TSNE(n_components=2, perplexity=30, n_iter=500, random_state=42)\
            .fit_transform(X_scaled[sample_idx])
plt.figure(figsize=(6,5))
sns.scatterplot(x=tsne_proj[:,0], y=tsne_proj[:,1], hue=y.iloc[sample_idx],
                alpha=0.5, palette=["#e74c3c","#2ecc71"])
plt.title("t-SNE: 2D Projection")
plt.xlabel("Dim1")
plt.ylabel("Dim2")
plt.tight_layout()
plt.savefig(f"{output_dir}/09_tsne_projection.png")
plt.close()

# ── 10) Random Forest feature importance (top 10) ─────────────────────────
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)
imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)
plt.figure(figsize=(8,6))
sns.barplot(x=imp.values, y=imp.index, palette="Blues_r")
plt.title("Top 10 Feature Importances (Random Forest)")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig(f"{output_dir}/10_rf_feature_importance.png")
plt.close()
