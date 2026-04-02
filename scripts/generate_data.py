"""
Gera dados ficticios de pacientes e registros financeiros.
Executar: python scripts/generate_data.py
"""

import csv
import random
import os
from datetime import datetime, timedelta

random.seed(42)

# Diretorio de saida
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# --- Configuracoes ---
CODIGOS_PACIENTES = [
    "ALPHA", "BETA", "GAMMA", "DELTA", "EPSILON", "ZETA", "ETA", "THETA",
    "IOTA", "KAPPA", "LAMBDA", "MU", "NU", "XI", "OMICRON", "PI", "RHO",
    "SIGMA", "TAU", "UPSILON", "PHI", "CHI", "PSI", "OMEGA", "ARES",
    "ATHENA", "HELIOS", "SELENE", "HERMES", "IRIS"
]

MODELOS_COBRANCA = ["avulso", "semanal", "quinzenal"]
VALORES_SESSAO = [150.0, 180.0, 200.0, 220.0, 250.0, 280.0, 300.0, 350.0, 400.0]

# Periodo: 6 meses (julho 2024 a dezembro 2024)
DATA_INICIO = datetime(2024, 7, 1)
DATA_FIM = datetime(2024, 12, 31)


def gerar_pacientes():
    """Gera tabela de pacientes (~30 pacientes)."""
    pacientes = []
    for i, codigo in enumerate(CODIGOS_PACIENTES, start=1):
        paciente = {
            "id_paciente": f"P{i:03d}",
            "nome_codigo": f"PAC-{codigo}",
            "valor_sessao": random.choice(VALORES_SESSAO),
            "modelo_cobranca": random.choice(MODELOS_COBRANCA),
        }
        pacientes.append(paciente)
    return pacientes


def gerar_datas_sessoes(modelo_cobranca, data_inicio, data_fim):
    """Gera datas de sessoes baseado no modelo de cobranca."""
    datas = []
    current = data_inicio

    if modelo_cobranca == "semanal":
        delta = timedelta(days=7)
    elif modelo_cobranca == "quinzenal":
        delta = timedelta(days=14)
    else:  # avulso
        delta = timedelta(days=random.randint(7, 21))

    while current <= data_fim:
        # Pular fins de semana (sessoes em dias uteis)
        if current.weekday() < 5:
            # Chance de faltar (10%)
            if random.random() > 0.10:
                datas.append(current)
        current += delta
        if modelo_cobranca == "avulso":
            delta = timedelta(days=random.randint(7, 21))

    return datas


def gerar_financeiro(pacientes):
    """Gera registros financeiros (~200 registros)."""
    registros = []
    id_registro = 1

    for paciente in pacientes:
        datas = gerar_datas_sessoes(
            paciente["modelo_cobranca"], DATA_INICIO, DATA_FIM
        )

        # Definir perfil de pagamento do paciente
        # ~75% bons pagadores, ~25% com atrasos frequentes
        taxa_inadimplencia = random.choice([0.05, 0.05, 0.05, 0.10, 0.10, 0.15, 0.30, 0.50])

        for data in datas:
            # Status de pagamento
            if random.random() < taxa_inadimplencia:
                status = "pendente"
            else:
                status = "pago"

            # NF emitida: quase sempre sim quando pago, raramente quando pendente
            if status == "pago":
                nf_emitida = random.random() < 0.95
            else:
                nf_emitida = random.random() < 0.10

            registro = {
                "id_registro": f"F{id_registro:03d}",
                "id_paciente": paciente["id_paciente"],
                "data_sessao": data.strftime("%Y-%m-%d"),
                "valor": paciente["valor_sessao"],
                "status_pagamento": status,
                "nf_emitida": str(nf_emitida).lower(),
            }
            registros.append(registro)
            id_registro += 1

    return registros


def salvar_csv(dados, nome_arquivo, campos):
    """Salva dados em CSV."""
    filepath = os.path.join(DATA_DIR, nome_arquivo)
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        writer.writerows(dados)
    print(f"  Salvo: {filepath} ({len(dados)} registros)")


def main():
    print("Gerando dados ficticios...\n")

    # Gerar pacientes
    pacientes = gerar_pacientes()
    salvar_csv(
        pacientes,
        "pacientes.csv",
        ["id_paciente", "nome_codigo", "valor_sessao", "modelo_cobranca"],
    )

    # Gerar financeiro
    financeiro = gerar_financeiro(pacientes)
    salvar_csv(
        financeiro,
        "financeiro.csv",
        ["id_registro", "id_paciente", "data_sessao", "valor", "status_pagamento", "nf_emitida"],
    )

    # Resumo
    pagos = sum(1 for r in financeiro if r["status_pagamento"] == "pago")
    pendentes = len(financeiro) - pagos
    print(f"\nResumo:")
    print(f"  Pacientes: {len(pacientes)}")
    print(f"  Registros financeiros: {len(financeiro)}")
    print(f"  Pagos: {pagos} ({pagos/len(financeiro)*100:.1f}%)")
    print(f"  Pendentes: {pendentes} ({pendentes/len(financeiro)*100:.1f}%)")


if __name__ == "__main__":
    main()
