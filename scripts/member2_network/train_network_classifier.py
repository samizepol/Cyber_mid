"""
train_network_classifier.py
Member 3 - Network-Traffic Classification (Step 2)

Baseline network classifier: random forest on all numeric network features, trained on a
stratified 75/25 train/test split, then saved to MODEL_OUTPUT. First experiment; the final
evaluation uses cross-validation (validate_network_model.py, train_network_compatible_api.py).

Usage:  python train_network_classifier.py
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Labelled dataset: the original file is not modified.
from config import DATASET, MODEL_OUTPUT

df = pd.read_csv(DATASET)

# Use only samples with a label and valid features.
df = df[df["family"].notna() & (df["status"] == "ok")].copy()

# Keep identifiers and labels out of the features.
exclude = {
    "analysis_id", "pcap_path", "sha256", "family",
    "label_agreement", "sample_id", "status"
}
features = [
    c for c in df.select_dtypes(include="number").columns
    if c not in exclude
]

X = df[features].replace([float("inf"), -float("inf")], float("nan"))
y = df["family"].astype(str)

print("Muestras:", len(df))
print("Características:", len(features))
print("\nMuestras por familia:")
print(y.value_counts())

# Hold out 25% of the data to evaluate the model.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Median imputation for missing values, scaling, random forest
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("classifier", RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42
    ))
])

model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("\nAccuracy:", round(accuracy_score(y_test, predictions), 4))
print("\nInforme de clasificación:")
print(classification_report(y_test, predictions, zero_division=0))

print("\nMatriz de confusión:")
print("Orden de familias:", sorted(y.unique()))
print(confusion_matrix(y_test, predictions, labels=sorted(y.unique())))

# Save the model without touching the original dataset.
import joblib

joblib.dump(
    {"model": model, "features": features},
    str(MODEL_OUTPUT)
)
print("\nModelo guardado en", MODEL_OUTPUT)
