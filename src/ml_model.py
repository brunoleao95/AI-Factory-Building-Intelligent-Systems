"""
Modelo ML: classificacao de risco de inadimplencia.
Usa scikit-learn para prever quais pacientes tem maior risco de atraso.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from src.database import get_connection


_model = None
_label_encoder = None
_features_df = None


def _extract_features():
    """
    Extrai features por paciente a partir dos dados financeiros.

    Returns:
        DataFrame com features e labels
    """
    conn = get_connection()

    # Dados por paciente
    df = conn.execute("""
        SELECT
            p.id_paciente,
            p.nome_codigo,
            p.valor_sessao,
            p.modelo_cobranca,
            COUNT(*) as total_sessoes,
            SUM(CASE WHEN f.status_pagamento = 'pendente' THEN 1 ELSE 0 END) as sessoes_pendentes,
            SUM(CASE WHEN f.nf_emitida = true THEN 1 ELSE 0 END) as nf_emitidas,
            AVG(f.valor) as valor_medio
        FROM pacientes p
        JOIN financeiro f ON p.id_paciente = f.id_paciente
        GROUP BY p.id_paciente, p.nome_codigo, p.valor_sessao, p.modelo_cobranca
    """).fetchdf()

    # Features derivadas
    df["taxa_atraso"] = df["sessoes_pendentes"] / df["total_sessoes"]
    df["taxa_nf"] = df["nf_emitidas"] / df["total_sessoes"]

    # Label: risco alto se taxa de atraso > 20%
    df["risco_alto"] = (df["taxa_atraso"] > 0.20).astype(int)

    return df


def train_model():
    """
    Treina modelo de classificacao de risco.

    Returns:
        Dict com metricas do modelo
    """
    global _model, _label_encoder, _features_df

    _features_df = _extract_features()

    # Encoder para modelo_cobranca
    _label_encoder = LabelEncoder()
    _features_df["modelo_encoded"] = _label_encoder.fit_transform(_features_df["modelo_cobranca"])

    # Features para o modelo
    feature_cols = ["total_sessoes", "taxa_atraso", "taxa_nf", "valor_sessao", "modelo_encoded"]
    X = _features_df[feature_cols].values
    y = _features_df["risco_alto"].values

    # Treinar RandomForest
    _model = RandomForestClassifier(
        n_estimators=50,
        max_depth=5,
        random_state=42,
        class_weight="balanced",
    )
    _model.fit(X, y)

    # Metricas basicas
    predictions = _model.predict(X)
    accuracy = (predictions == y).mean()

    n_alto = int(y.sum())
    n_baixo = int(len(y) - y.sum())

    return {
        "accuracy": round(accuracy, 3),
        "total_pacientes": len(y),
        "risco_alto": n_alto,
        "risco_baixo": n_baixo,
        "features": feature_cols,
    }


def predict_risk(id_paciente=None):
    """
    Prediz risco de inadimplencia para um paciente ou todos.

    Args:
        id_paciente: ID do paciente (ex: P001). Se None, retorna todos.

    Returns:
        DataFrame com id_paciente, nome_codigo, risco, probabilidade
    """
    global _model, _features_df

    if _model is None:
        train_model()

    feature_cols = ["total_sessoes", "taxa_atraso", "taxa_nf", "valor_sessao", "modelo_encoded"]

    if id_paciente:
        df = _features_df[_features_df["id_paciente"] == id_paciente].copy()
        if df.empty:
            return None
    else:
        df = _features_df.copy()

    X = df[feature_cols].values
    probabilities = _model.predict_proba(X)

    # Probabilidade de risco alto (classe 1)
    df = df.copy()
    if probabilities.shape[1] > 1:
        df["probabilidade_risco"] = probabilities[:, 1]
    else:
        df["probabilidade_risco"] = probabilities[:, 0]

    df["risco"] = df["probabilidade_risco"].apply(
        lambda p: "ALTO" if p >= 0.5 else ("MEDIO" if p >= 0.3 else "BAIXO")
    )

    result = df[["id_paciente", "nome_codigo", "taxa_atraso", "sessoes_pendentes",
                  "total_sessoes", "risco", "probabilidade_risco"]].copy()
    result = result.sort_values("probabilidade_risco", ascending=False)

    return result


def get_risk_summary():
    """Retorna resumo de risco de todos os pacientes."""
    risks = predict_risk()
    if risks is None:
        return "Nao foi possivel calcular riscos."

    alto = risks[risks["risco"] == "ALTO"]
    medio = risks[risks["risco"] == "MEDIO"]
    baixo = risks[risks["risco"] == "BAIXO"]

    summary = f"**Analise de Risco de Inadimplencia**\n\n"
    summary += f"- 🔴 Risco Alto: {len(alto)} pacientes\n"
    summary += f"- 🟡 Risco Medio: {len(medio)} pacientes\n"
    summary += f"- 🟢 Risco Baixo: {len(baixo)} pacientes\n\n"

    if not alto.empty:
        summary += "**Pacientes com Risco Alto:**\n\n"
        summary += "| Paciente | Taxa Atraso | Sessoes Pendentes | Probabilidade |\n"
        summary += "|----------|-------------|-------------------|---------------|\n"
        for _, row in alto.iterrows():
            summary += f"| {row['nome_codigo']} | {row['taxa_atraso']:.0%} | {int(row['sessoes_pendentes'])} | {row['probabilidade_risco']:.0%} |\n"

    return summary
