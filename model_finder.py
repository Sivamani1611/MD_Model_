import os  # To handle directory creation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA

# Load the dataset
dataset = pd.read_csv('./datasets/dataset.csv', sep=',', low_memory=False)

# Save count visualization
output_dir = "visualizations"
os.makedirs(output_dir, exist_ok=True)

malicious_counts = dataset['legitimate'].value_counts()
labels = ["Non-Malicious", "Malicious"]

plt.figure(figsize=(6, 4))
sns.barplot(x=labels, y=malicious_counts, palette="viridis")
plt.title("Count of Malicious and Non-Malicious Files")
plt.xlabel("File Type")
plt.ylabel("Count")
image_path_counts = os.path.join(output_dir, "malicious_non_malicious_counts.png")
plt.savefig(image_path_counts)
plt.close()
print(f"Malicious/Non-Malicious count image saved as: {image_path_counts}")

# Preprocess the data
X = dataset.drop(['ID', 'md5', 'legitimate'], axis=1).values
y = dataset['legitimate'].values

# Feature selection using ExtraTreesClassifier
extratrees = ExtraTreesClassifier().fit(X, y)
model = SelectFromModel(extratrees, prefit=True)
X_new = model.transform(X)
nbfeatures = X_new.shape[1]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=42)

# Ensure the directory for confusion matrices exists
output_dir_cm = "confusion_matrices"
os.makedirs(output_dir_cm, exist_ok=True)

# Dictionary of ML models to analyze
models = {
    "Random Forest": RandomForestClassifier(n_estimators=33, random_state=42),
    "Bagging": BaggingClassifier(n_estimators=50, random_state=42),
    "Logistic Regression (Multinomial)": LogisticRegression(max_iter=1000, multi_class='multinomial', random_state=42),
    "Logistic Regression (Binary)": LogisticRegression(max_iter=1000, random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Decision Tree (Pruned)": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Naive Bayes (Gaussian)": GaussianNB(),
    "Perceptron": Perceptron(max_iter=1000, random_state=42),
    "LDA (Linear Discriminant Analysis)": LDA(),
    "QDA (Quadratic Discriminant Analysis)": QDA(),
    "AdaBoost": AdaBoostClassifier(n_estimators=50, random_state=42),
}

# Analyze models and collect metrics
results = []
for model_name, model in models.items():
    print(f"Training {model_name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    # Generate confusion matrix and save as an image
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Non-legitimate", "Legitimate"], yticklabels=["Non-legitimate", "Legitimate"])
    plt.title(f'Confusion Matrix: {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    # Save confusion matrix image
    image_path_cm = os.path.join(output_dir_cm, f"{model_name.replace(' ', '_')}_confusion_matrix.png")
    plt.savefig(image_path_cm)
    plt.close()
    print(f"Confusion matrix saved as: {image_path_cm}")
    
    # Append to results
    results.append({"Model": model_name, "Accuracy": accuracy, "F1-Score": f1, "Precision": precision, "Recall": recall})

# Convert results to a DataFrame for tabular display
results_df = pd.DataFrame(results)
print("\nModel Performance Summary:")
print(results_df)
