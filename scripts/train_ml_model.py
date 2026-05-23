"""
Etapa 2 - treino do modelo de risco com FLAML AutoML sobre dataset publico UCI.

Dataset: 'Default of Credit Card Clients' (UCI ID 350, ~30k registros).
Contextualizado para inadimplencia em consultorio de psicologia: o problema
estrutural e o mesmo (prever quem deixara de pagar entre uma serie de
pagamentos esperados).

Uso:
    python scripts/train_ml_model.py

Saida:
    models/risco_inadimplencia.pkl  - modelo FLAML serializado
    models/metrics.json             - metricas (accuracy, precision, recall, F1, ROC-AUC)
    data/uci_credit_default.csv     - cache local do dataset (gitignored)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from flaml import AutoML
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
CACHED_CSV = DATA_DIR / "uci_credit_default.csv"
MODEL_PATH = MODELS_DIR / "risco_inadimplencia.pkl"
METRICS_PATH = MODELS_DIR / "metrics.json"

# Features que serao usadas para previsao (subset do dataset UCI).
# Mapeamento contextual:
#   LIMIT_BAL  -> "valor de credito" (analogo a soma esperada de sessoes)
#   PAY_0..6   -> historico de atrasos (analogo a taxa_atraso historica)
#   BILL_AMT_X -> valor faturado (analogo a valor por sessao)
#   PAY_AMT_X  -> valor pago (analogo a recebido)
FEATURE_COLS = [
    "LIMIT_BAL",
    "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3",
]
TARGET_COL = "default_payment_next_month"

# Tempo de busca do FLAML (segundos). 60s e suficiente neste dataset.
TIME_BUDGET = 60


# Mapeamento dos nomes genericos do UCI (X1..X23) para os nomes canonicos
# do dataset 'default of credit card clients' (Yeh & Lien, 2009).
UCI_RENAME = {
    "X1": "LIMIT_BAL", "X2": "SEX", "X3": "EDUCATION",
    "X4": "MARRIAGE", "X5": "AGE",
    "X6": "PAY_0", "X7": "PAY_2", "X8": "PAY_3",
    "X9": "PAY_4", "X10": "PAY_5", "X11": "PAY_6",
    "X12": "BILL_AMT1", "X13": "BILL_AMT2", "X14": "BILL_AMT3",
    "X15": "BILL_AMT4", "X16": "BILL_AMT5", "X17": "BILL_AMT6",
    "X18": "PAY_AMT1", "X19": "PAY_AMT2", "X20": "PAY_AMT3",
    "X21": "PAY_AMT4", "X22": "PAY_AMT5", "X23": "PAY_AMT6",
}


def fetch_dataset() -> pd.DataFrame:
    """
    Baixa o dataset UCI na primeira execucao e cacheia localmente.
    Usa o pacote oficial `ucimlrepo`.
    """
    if CACHED_CSV.exists():
        print(f"[cache] dataset ja em {CACHED_CSV}")
        return pd.read_csv(CACHED_CSV)

    print("[download] baixando UCI Default of Credit Card Clients (id=350)...")
    from ucimlrepo import fetch_ucirepo  # importacao lazy

    bundle = fetch_ucirepo(id=350)
    X = bundle.data.features
    y = bundle.data.targets

    target_name = y.columns[0]
    df = pd.concat([X, y], axis=1).rename(columns={target_name: TARGET_COL})
    df = df.rename(columns=UCI_RENAME)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CACHED_CSV, index=False)
    print(f"[ok] cacheado em {CACHED_CSV} ({len(df)} registros)")
    return df


def train(df: pd.DataFrame) -> dict:
    """Treina FLAML, avalia no test set e salva artefatos."""
    missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing:
        raise RuntimeError(f"Colunas faltando no dataset: {missing}")

    X = df[FEATURE_COLS].astype(float)
    y = df[TARGET_COL].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42,
    )

    automl = AutoML()
    automl.fit(
        X_train=X_train,
        y_train=y_train,
        task="classification",
        metric="roc_auc",
        time_budget=TIME_BUDGET,
        estimator_list=["lgbm", "rf", "xgboost", "extra_tree"],
        eval_method="cv",
        n_splits=3,
        log_file_name="",
        verbose=1,
    )

    print(f"\n[flaml] melhor estimador: {automl.best_estimator}")
    print(f"[flaml] melhor configuracao: {automl.best_config}")

    y_pred = automl.predict(X_test)
    y_proba = automl.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred)), 4),
        "recall": round(float(recall_score(y_test, y_pred)), 4),
        "f1": round(float(f1_score(y_test, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
        "best_estimator": automl.best_estimator,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "n_features": len(FEATURE_COLS),
        "feature_cols": FEATURE_COLS,
        "target_col": TARGET_COL,
        "time_budget_s": TIME_BUDGET,
        "dataset": "UCI Default of Credit Card Clients (id=350)",
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(automl, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\n[ok] modelo salvo em {MODEL_PATH}")
    print(f"[ok] metricas em {METRICS_PATH}")

    return metrics


def main() -> int:
    df = fetch_dataset()
    metrics = train(df)

    print("\n=== Metricas no conjunto de teste ===")
    for k in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        print(f"  {k:10s}: {metrics[k]:.4f}")
    print(f"  estimator : {metrics['best_estimator']}")
    print(f"  treino    : {metrics['n_train']} | teste: {metrics['n_test']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
