"""
Modelo ML: classificacao de risco de inadimplencia.

Etapa 2: o modelo FLAML/XGBoost e treinado offline pelo script
`scripts/train_ml_model.py` sobre o dataset publico UCI 'Default of Credit
Card Clients'. Aqui carregamos o artefato e mapeamos as features de cada
paciente do DuckDB para o espaco de features que o modelo conhece.

Compatibilidade com a etapa 1: as funcoes `train_model()`, `predict_risk()`
e `get_risk_summary()` mantem assinaturas e colunas de retorno usadas por
`src/agents.py`.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.database import get_connection

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "risco_inadimplencia.pkl"
METRICS_PATH = ROOT / "models" / "metrics.json"

# Features que o modelo FLAML espera (mesma lista do script de treino)
FEATURE_COLS = [
    "LIMIT_BAL",
    "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3",
]

_model = None
_metrics: dict | None = None
_features_df: pd.DataFrame | None = None


def _load_artifacts():
    """Carrega modelo FLAML e metricas. Lanca se nao existirem."""
    global _model, _metrics
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Modelo nao encontrado em {MODEL_PATH}. "
                "Rode primeiro: python scripts/train_ml_model.py"
            )
        _model = joblib.load(MODEL_PATH)
    if _metrics is None and METRICS_PATH.exists():
        import json
        _metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    return _model, _metrics


def _extract_patient_features() -> pd.DataFrame:
    """
    Para cada paciente do DuckDB, calcula:
      - features descritivas (taxa_atraso, sessoes_pendentes, total_sessoes,
        nome_codigo, valor_sessao) — usadas pela UI;
      - vetor de features no espaco UCI — usado pelo modelo.

    Mapeamento contextual (UCI -> psicologia):
      LIMIT_BAL  : total esperado (total_sessoes * valor_sessao)
      AGE        : 35 (mediana de adultos em terapia, nao temos a coluna)
      PAY_0..6   : status de atraso nos ultimos 6 meses agregados (0 = em
                   dia, 1+ = atrasos crescentes, conforme convencao UCI)
      BILL_AMT_X : valor faturado em cada um dos ultimos 6 meses
      PAY_AMT_X  : valor pago em cada um dos ultimos 6 meses
    """
    conn = get_connection()

    base = conn.execute("""
        SELECT
            p.id_paciente,
            p.nome_codigo,
            p.valor_sessao,
            p.modelo_cobranca,
            COUNT(*) as total_sessoes,
            SUM(CASE WHEN f.status_pagamento = 'pendente' THEN 1 ELSE 0 END) as sessoes_pendentes,
            SUM(CASE WHEN f.nf_emitida = true THEN 1 ELSE 0 END) as nf_emitidas
        FROM pacientes p
        JOIN financeiro f ON p.id_paciente = f.id_paciente
        GROUP BY p.id_paciente, p.nome_codigo, p.valor_sessao, p.modelo_cobranca
    """).fetchdf()

    base["taxa_atraso"] = base["sessoes_pendentes"] / base["total_sessoes"]
    base["taxa_nf"] = base["nf_emitidas"] / base["total_sessoes"]

    monthly = conn.execute("""
        SELECT
            id_paciente,
            DATE_TRUNC('month', data_sessao) AS mes,
            SUM(valor) AS valor_total,
            SUM(CASE WHEN status_pagamento = 'pago' THEN valor ELSE 0 END) AS valor_pago,
            SUM(CASE WHEN status_pagamento = 'pendente' THEN 1 ELSE 0 END) AS pendentes
        FROM financeiro
        GROUP BY id_paciente, DATE_TRUNC('month', data_sessao)
        ORDER BY id_paciente, mes DESC
    """).fetchdf()

    rows: list[dict] = []
    for _, p in base.iterrows():
        last6 = monthly[monthly["id_paciente"] == p["id_paciente"]].head(6)
        bill_amts = list(last6["valor_total"].values) + [0.0] * 6
        pay_amts = list(last6["valor_pago"].values) + [0.0] * 6
        pay_status = list(last6["pendentes"].values) + [0] * 6

        rows.append({
            "id_paciente": p["id_paciente"],
            "nome_codigo": p["nome_codigo"],
            "valor_sessao": p["valor_sessao"],
            "modelo_cobranca": p["modelo_cobranca"],
            "total_sessoes": int(p["total_sessoes"]),
            "sessoes_pendentes": int(p["sessoes_pendentes"]),
            "taxa_atraso": float(p["taxa_atraso"]),
            "taxa_nf": float(p["taxa_nf"]),
            "LIMIT_BAL": float(p["total_sessoes"] * p["valor_sessao"]),
            "AGE": 35.0,
            "PAY_0": int(min(pay_status[0], 8)),
            "PAY_2": int(min(pay_status[1], 8)),
            "PAY_3": int(min(pay_status[2], 8)),
            "PAY_4": int(min(pay_status[3], 8)),
            "PAY_5": int(min(pay_status[4], 8)),
            "PAY_6": int(min(pay_status[5], 8)),
            "BILL_AMT1": float(bill_amts[0]),
            "BILL_AMT2": float(bill_amts[1]),
            "BILL_AMT3": float(bill_amts[2]),
            "PAY_AMT1": float(pay_amts[0]),
            "PAY_AMT2": float(pay_amts[1]),
            "PAY_AMT3": float(pay_amts[2]),
        })

    return pd.DataFrame(rows)


def train_model() -> dict:
    """
    Etapa 2: nao treina mais aqui — apenas carrega o modelo FLAML serializado
    e o cache de metricas geradas pelo script offline. Retorna dict no mesmo
    formato esperado pela UI da etapa 1.
    """
    global _features_df
    model, metrics = _load_artifacts()
    _features_df = _extract_patient_features()

    n_total = len(_features_df)
    # Aplica o modelo aos pacientes para contar quantos seriam classificados como alto risco
    if n_total:
        proba = model.predict_proba(_features_df[FEATURE_COLS].values)[:, 1]
        n_alto = int((proba >= 0.5).sum())
    else:
        n_alto = 0

    summary = {
        "accuracy": metrics.get("accuracy") if metrics else None,
        "precision": metrics.get("precision") if metrics else None,
        "recall": metrics.get("recall") if metrics else None,
        "f1": metrics.get("f1") if metrics else None,
        "roc_auc": metrics.get("roc_auc") if metrics else None,
        "best_estimator": metrics.get("best_estimator") if metrics else None,
        "total_pacientes": n_total,
        "risco_alto": n_alto,
        "risco_baixo": n_total - n_alto,
        "features": FEATURE_COLS,
        "dataset": metrics.get("dataset") if metrics else None,
    }
    return summary


def predict_risk(id_paciente: str | None = None):
    """
    Prediz risco de inadimplencia para um paciente ou todos.

    Args:
        id_paciente: ID interno (P001 etc). Se None, retorna todos.

    Returns:
        DataFrame com id_paciente, nome_codigo, taxa_atraso, sessoes_pendentes,
        total_sessoes, risco, probabilidade_risco. Mesmas colunas da etapa 1.
    """
    global _features_df
    model, _ = _load_artifacts()

    if _features_df is None:
        train_model()

    if id_paciente:
        df = _features_df[_features_df["id_paciente"] == id_paciente].copy()
        if df.empty:
            return None
    else:
        df = _features_df.copy()

    X = df[FEATURE_COLS].values
    probabilities = model.predict_proba(X)
    df["probabilidade_risco"] = probabilities[:, 1] if probabilities.shape[1] > 1 else probabilities[:, 0]

    df["risco"] = df["probabilidade_risco"].apply(
        lambda p: "ALTO" if p >= 0.5 else ("MEDIO" if p >= 0.3 else "BAIXO")
    )

    out = df[[
        "id_paciente", "nome_codigo", "taxa_atraso", "sessoes_pendentes",
        "total_sessoes", "risco", "probabilidade_risco",
    ]].copy()
    return out.sort_values("probabilidade_risco", ascending=False)


def get_risk_summary() -> str:
    """Retorna resumo formatado em markdown do risco de todos os pacientes."""
    risks = predict_risk()
    if risks is None or risks.empty:
        return "Nao foi possivel calcular riscos."

    alto = risks[risks["risco"] == "ALTO"]
    medio = risks[risks["risco"] == "MEDIO"]
    baixo = risks[risks["risco"] == "BAIXO"]

    out = "**Analise de Risco de Inadimplencia**\n\n"
    out += f"- Risco Alto: {len(alto)} pacientes\n"
    out += f"- Risco Medio: {len(medio)} pacientes\n"
    out += f"- Risco Baixo: {len(baixo)} pacientes\n\n"

    if not alto.empty:
        out += "**Pacientes com Risco Alto:**\n\n"
        out += "| Paciente | Taxa Atraso | Sessoes Pendentes | Probabilidade |\n"
        out += "|----------|-------------|-------------------|---------------|\n"
        for _, row in alto.iterrows():
            out += (
                f"| {row['nome_codigo']} | {row['taxa_atraso']:.0%} | "
                f"{int(row['sessoes_pendentes'])} | {row['probabilidade_risco']:.0%} |\n"
            )

    return out


def get_metrics() -> dict | None:
    """Expoe as metricas treinadas (usado pela sidebar e pelo relatorio)."""
    _, metrics = _load_artifacts()
    return metrics
