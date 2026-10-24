
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from config import DATASET, CV_RESULTS

# Cargar el dataset original sin modificarlo.
df = pd.read_csv(DATASET)
df = df[(df["status"] == "ok") & df["family"].notna()].copy()

# Usar solo características numéricas; excluir identificadores y etiquetas.
excluded = {
    "analysis_id", "pcap_path", "sha256", "family",
    "label_agreement", "sample_id", "status"
}
features = [
    col for col in df.select_dtypes(include="number").columns
    if col not in excluded
]

X = df[features].replace([np.inf, -np.inf], np.nan)
y = df["family"].astype(str)

print("Total de muestras:", len(df))
print("Características:", len(features))
print("\nMuestras por familia:")
print(y.value_counts())

# Cinco particiones estratificadas.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Imputación y escalado dentro de cada partición para evitar fugas.
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("classifier", RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42
    ))
])

metrics = {
    "accuracy": "accuracy",
    "f1_macro": "f1_macro",
    "precision_macro": "precision_macro",
    "recall_macro": "recall_macro"
}

results = cross_validate(
    model, X, y,
    cv=cv,
    scoring=metrics,
    return_train_score=False,
    error_score="raise"
)

print("\n=== RESULTADOS DE VALIDACIÓN CRUZADA ===")
for metric in metrics:
    scores = results["test_" + metric]
    print(
        "{}: {} ± {} | particiones: {}".format(
            metric,
            round(scores.mean(), 4),
            round(scores.std(), 4),
            [round(s, 4) for s in scores]
        )
    )

# Guardar las métricas para documentar el experimento.
summary = pd.DataFrame({
    "fold": range(1, 6),
    "accuracy": results["test_accuracy"],
    "f1_macro": results["test_f1_macro"],
    "precision_macro": results["test_precision_macro"],
    "recall_macro": results["test_recall_macro"]
})

summary.to_csv(CV_RESULTS, index=False)
print("\nResultados guardados en", CV_RESULTS)
