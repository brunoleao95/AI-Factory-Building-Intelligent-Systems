"""
DuckDB: carregar CSVs, executar queries SQL, obter schema.
"""

import os
import duckdb
import pandas as pd
from config import DATA_DIR

# Conexao global in-memory
_conn = None


def get_connection():
    """Retorna conexao DuckDB (singleton)."""
    global _conn
    if _conn is None:
        _conn = duckdb.connect(database=":memory:")
    return _conn


def load_tables():
    """
    Carrega CSVs da pasta data/ como tabelas no DuckDB.
    Cria as tabelas 'pacientes' e 'financeiro'.
    """
    conn = get_connection()

    pacientes_path = os.path.join(DATA_DIR, "pacientes.csv")
    financeiro_path = os.path.join(DATA_DIR, "financeiro.csv")

    if not os.path.exists(pacientes_path):
        raise FileNotFoundError(f"Arquivo nao encontrado: {pacientes_path}")
    if not os.path.exists(financeiro_path):
        raise FileNotFoundError(f"Arquivo nao encontrado: {financeiro_path}")

    # Dropar tabelas se existirem (para recarregar)
    conn.execute("DROP TABLE IF EXISTS pacientes")
    conn.execute("DROP TABLE IF EXISTS financeiro")

    # Carregar CSVs
    conn.execute(f"CREATE TABLE pacientes AS SELECT * FROM read_csv_auto('{pacientes_path.replace(chr(92), '/')}')")
    conn.execute(f"CREATE TABLE financeiro AS SELECT * FROM read_csv_auto('{financeiro_path.replace(chr(92), '/')}')")

    # Verificar
    n_pacientes = conn.execute("SELECT COUNT(*) FROM pacientes").fetchone()[0]
    n_financeiro = conn.execute("SELECT COUNT(*) FROM financeiro").fetchone()[0]

    return {
        "pacientes": n_pacientes,
        "financeiro": n_financeiro,
    }


def execute_query(sql):
    """
    Executa query SQL no DuckDB e retorna DataFrame.

    Args:
        sql: Query SQL valida

    Returns:
        pandas DataFrame com os resultados

    Raises:
        Exception se a query for invalida ou perigosa
    """
    conn = get_connection()

    # Seguranca: bloquear operacoes destrutivas
    sql_upper = sql.upper().strip()
    blocked = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE", "CREATE"]
    for keyword in blocked:
        if sql_upper.startswith(keyword):
            raise ValueError(f"Operacao '{keyword}' nao permitida.")

    result = conn.execute(sql).fetchdf()
    return result


def get_schema_description():
    """
    Retorna descricao textual das tabelas para o LLM usar no Text-to-SQL.

    Returns:
        String com schema descritivo
    """
    return """
TABELAS DISPONIVEIS:

1. Tabela: pacientes
   Colunas:
   - id_paciente (VARCHAR): Identificador unico do paciente (ex: P001, P002)
   - nome_codigo (VARCHAR): Codigo anonimizado do paciente (ex: PAC-ALPHA, PAC-BETA)
   - valor_sessao (DOUBLE): Valor cobrado por sessao em reais (ex: 250.00)
   - modelo_cobranca (VARCHAR): Modelo de cobranca - valores: 'avulso', 'semanal', 'quinzenal'

2. Tabela: financeiro
   Colunas:
   - id_registro (VARCHAR): Identificador unico do registro (ex: F001, F002)
   - id_paciente (VARCHAR): Referencia ao paciente (FK para pacientes.id_paciente)
   - data_sessao (DATE): Data da sessao (formato YYYY-MM-DD)
   - valor (DOUBLE): Valor da sessao em reais
   - status_pagamento (VARCHAR): Status - valores: 'pago', 'pendente'
   - nf_emitida (BOOLEAN): Se a nota fiscal foi emitida (true/false)

RELACIONAMENTO: financeiro.id_paciente -> pacientes.id_paciente (JOIN)

EXEMPLOS DE QUERIES:
- Total faturado: SELECT SUM(valor) FROM financeiro WHERE status_pagamento = 'pago'
- Pacientes inadimplentes: SELECT p.nome_codigo, COUNT(*) as pendentes FROM financeiro f JOIN pacientes p ON f.id_paciente = p.id_paciente WHERE f.status_pagamento = 'pendente' GROUP BY p.nome_codigo ORDER BY pendentes DESC
- Faturamento mensal: SELECT strftime(data_sessao, '%Y-%m') as mes, SUM(valor) as total FROM financeiro WHERE status_pagamento = 'pago' GROUP BY mes ORDER BY mes
"""


def get_summary():
    """
    Retorna resumo rapido dos dados carregados.

    Returns:
        Dict com estatisticas basicas
    """
    conn = get_connection()

    try:
        total_pacientes = conn.execute("SELECT COUNT(*) FROM pacientes").fetchone()[0]
        total_registros = conn.execute("SELECT COUNT(*) FROM financeiro").fetchone()[0]
        total_pago = conn.execute("SELECT SUM(valor) FROM financeiro WHERE status_pagamento = 'pago'").fetchone()[0]
        total_pendente = conn.execute("SELECT SUM(valor) FROM financeiro WHERE status_pagamento = 'pendente'").fetchone()[0]
        pendentes_count = conn.execute("SELECT COUNT(*) FROM financeiro WHERE status_pagamento = 'pendente'").fetchone()[0]

        return {
            "total_pacientes": total_pacientes,
            "total_registros": total_registros,
            "total_pago": total_pago or 0,
            "total_pendente": total_pendente or 0,
            "pendentes_count": pendentes_count,
        }
    except Exception:
        return None
