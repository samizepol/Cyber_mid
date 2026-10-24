
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Dataset etiquetado: no modifica el archivo original.
from config import DATASET, MODEL_OUTPUT

df = pd.read_csv(DATASET)

# Usar únicamente muestras con etiqueta y características válidas.
df = df[df["family"].notna() & (df["status"] == "ok")].copy()

# Evitar que identificadores y etiquetas se conviertan en características.
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

# Reservar datos para evaluar el modelo.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

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

# Guardar el modelo y el informe sin tocar el dataset original.
import joblib

joblib.dump(
    {"model": model, "features": features},
    str(MODEL_OUTPUT)
)
print("\nModelo guardado en", MODEL_OUTPUT)
