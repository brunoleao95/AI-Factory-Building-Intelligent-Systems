"""
Agentes: roteamento de perguntas e processamento especializado.
- Agente Financeiro: consultas DuckDB via Text-to-SQL
- Agente Repositorio: busca semantica via RAG
- Roteador: detecta intencao e despacha para o agente correto
"""

from src.database import execute_query, get_schema_description
from src.rag import search, is_indexed
from src.llm import chat_stream, text_to_sql, rag_answer, generate_collection_message, chat
from src.ml_model import predict_risk, get_risk_summary


# Palavras-chave para roteamento
KEYWORDS_FINANCEIRO = [
    "financeiro", "pagamento", "pago", "pendente", "faturamento", "receita",
    "inadimplencia", "inadimplente", "nota fiscal", "nf", "cobranca", "cobrar",
    "valor", "sessao", "sessoes", "paciente", "quanto", "total", "mensal",
    "mes", "dinheiro", "receber", "recebido", "fatura", "boleto", "whatsapp",
]

KEYWORDS_RAG = [
    "documento", "documentos", "artigo", "dsm", "cid", "tecnica", "tcc",
    "terapia", "cognitivo", "comportamental", "etica", "codigo de etica",
    "cfp", "conselho", "manual", "protocolo", "pesquisa", "estudo",
    "referencia", "bibliografia", "teoria", "abordagem", "diagnostico",
]

KEYWORDS_RISCO = [
    "risco", "inadimplencia", "classificacao", "prever", "previsao",
    "probabilidade", "ml", "modelo", "machine learning", "predizer",
]


def route_question(question):
    """
    Detecta intencao da pergunta e retorna o tipo de agente.

    Args:
        question: Pergunta do usuario

    Returns:
        String: 'financeiro', 'rag', 'risco', ou 'geral'
    """
    q_lower = question.lower()

    # Verificar cobranca WhatsApp (financeiro + geracao de mensagem)
    if any(w in q_lower for w in ["cobranca", "cobrar", "whatsapp", "mensagem"]):
        if any(w in q_lower for w in ["gerar", "criar", "escrever", "fazer", "enviar"]):
            return "cobranca"

    # Verificar risco
    risco_score = sum(1 for kw in KEYWORDS_RISCO if kw in q_lower)
    if risco_score >= 1:
        return "risco"

    # Verificar financeiro
    fin_score = sum(1 for kw in KEYWORDS_FINANCEIRO if kw in q_lower)

    # Verificar RAG
    rag_score = sum(1 for kw in KEYWORDS_RAG if kw in q_lower)

    if fin_score > rag_score and fin_score > 0:
        return "financeiro"
    elif rag_score > fin_score and rag_score > 0:
        return "rag"
    elif fin_score > 0:
        return "financeiro"
    elif rag_score > 0:
        return "rag"

    return "geral"


def agent_financeiro(question):
    """
    Agente Financeiro: converte pergunta em SQL, executa e formata resposta.

    Args:
        question: Pergunta do usuario sobre financas

    Yields:
        Tokens da resposta (streaming)
    """
    schema = get_schema_description()

    # Converter para SQL
    sql = text_to_sql(question, schema)

    if not sql or sql.startswith("Erro"):
        yield "Desculpe, nao consegui gerar a consulta. Tente reformular a pergunta."
        return

    # Executar query
    try:
        df = execute_query(sql)
    except ValueError as e:
        yield f"Operacao bloqueada: {str(e)}"
        return
    except Exception as e:
        yield f"Erro na consulta SQL: {str(e)}\n\nSQL gerada: `{sql}`\n\nTente reformular a pergunta."
        return

    if df.empty:
        yield "A consulta nao retornou resultados. Tente reformular a pergunta."
        return

    # Formatar resposta com LLM
    data_str = df.to_string(index=False)
    prompt = f"""O usuario perguntou: "{question}"

A consulta SQL executada foi: {sql}

Resultados obtidos:
{data_str}

Formate uma resposta clara e profissional com base nesses dados.
- Use R$ para valores monetarios
- Use tabelas markdown quando apropriado
- Seja objetiva e util"""

    messages = [{"role": "user", "content": prompt}]
    yield from chat_stream(messages)


def agent_repositorio(question):
    """
    Agente Repositorio: busca nos documentos indexados e responde via RAG.

    Args:
        question: Pergunta sobre documentos tecnicos

    Yields:
        Tokens da resposta (streaming)
    """
    if not is_indexed():
        yield "Nenhum documento foi indexado ainda. Coloque arquivos (.txt ou .pdf) na pasta `docs/` e reinicie o aplicativo."
        return

    # Buscar chunks relevantes
    results = search(question)

    if not results:
        yield "Nao encontrei informacoes relevantes nos documentos indexados. Tente reformular a pergunta."
        return

    # Montar contexto com fonte
    context_chunks = []
    for r in results:
        chunk_with_source = f"[Fonte: {r['source']}]\n{r['text']}"
        context_chunks.append(chunk_with_source)

    # Gerar resposta RAG com streaming
    yield from rag_answer(question, context_chunks)


def agent_risco(question):
    """
    Agente de Risco: analisa risco de inadimplencia.

    Args:
        question: Pergunta sobre risco

    Yields:
        Tokens da resposta (streaming)
    """
    q_lower = question.lower()

    # Verificar se e sobre um paciente especifico
    import re
    paciente_match = re.search(r'(p\d{3}|pac-\w+)', q_lower)

    if paciente_match:
        code = paciente_match.group(1).upper()
        if code.startswith("PAC-"):
            # Buscar id_paciente pelo nome_codigo
            from src.database import execute_query
            try:
                df = execute_query(f"SELECT id_paciente FROM pacientes WHERE nome_codigo = '{code}'")
                if not df.empty:
                    code = df.iloc[0]["id_paciente"]
            except Exception:
                pass

        risk = predict_risk(code)
        if risk is not None and not risk.empty:
            row = risk.iloc[0]
            yield f"**Analise de Risco - {row['nome_codigo']}**\n\n"
            yield f"- Risco: **{row['risco']}**\n"
            yield f"- Probabilidade: {row['probabilidade_risco']:.0%}\n"
            yield f"- Taxa de atraso: {row['taxa_atraso']:.0%}\n"
            yield f"- Sessoes pendentes: {int(row['sessoes_pendentes'])} de {int(row['total_sessoes'])}\n"
        else:
            yield f"Paciente {code} nao encontrado."
    else:
        # Resumo geral
        summary = get_risk_summary()
        yield summary


def agent_cobranca(question):
    """
    Agente de Cobranca: gera mensagem de cobranca para WhatsApp.

    Yields:
        Tokens da resposta
    """
    # Buscar pacientes com pendencias
    from src.database import execute_query
    try:
        df = execute_query("""
            SELECT p.nome_codigo, p.id_paciente,
                   COUNT(*) as sessoes_pendentes,
                   SUM(f.valor) as valor_total
            FROM financeiro f
            JOIN pacientes p ON f.id_paciente = p.id_paciente
            WHERE f.status_pagamento = 'pendente'
            GROUP BY p.nome_codigo, p.id_paciente
            ORDER BY valor_total DESC
        """)
    except Exception:
        yield "Erro ao consultar pacientes com pendencias."
        return

    if df.empty:
        yield "Nao ha pacientes com pagamentos pendentes no momento."
        return

    # Verificar se mencionou paciente especifico
    import re
    q_lower = question.lower()
    paciente_match = re.search(r'pac-(\w+)', q_lower)

    if paciente_match:
        code = f"PAC-{paciente_match.group(1).upper()}"
        row = df[df["nome_codigo"] == code]
        if row.empty:
            yield f"Paciente {code} nao possui pagamentos pendentes."
            return
        row = row.iloc[0]
        msg = generate_collection_message(row["nome_codigo"], row["valor_total"], row["sessoes_pendentes"])
        yield f"**Mensagem de cobranca para {row['nome_codigo']}:**\n\n{msg}"
    else:
        # Mostrar lista de pendentes e gerar para o primeiro
        yield "**Pacientes com pagamentos pendentes:**\n\n"
        yield "| Paciente | Sessoes Pendentes | Valor Total |\n"
        yield "|----------|-------------------|-------------|\n"
        for _, row in df.iterrows():
            yield f"| {row['nome_codigo']} | {int(row['sessoes_pendentes'])} | R$ {row['valor_total']:.2f} |\n"
        yield "\nPara gerar uma mensagem de cobranca, diga: *\"Gerar cobranca para PAC-NOME\"*"


def process_question(question):
    """
    Processa pergunta do usuario: roteia e executa agente adequado.

    Args:
        question: Pergunta do usuario

    Returns:
        Tuple (tipo_agente, generator_de_tokens)
    """
    agent_type = route_question(question)

    if agent_type == "financeiro":
        return agent_type, agent_financeiro(question)
    elif agent_type == "rag":
        return agent_type, agent_repositorio(question)
    elif agent_type == "risco":
        return agent_type, agent_risco(question)
    elif agent_type == "cobranca":
        return agent_type, agent_cobranca(question)
    else:
        # Resposta geral via LLM
        messages = [{"role": "user", "content": question}]
        return agent_type, chat_stream(messages)
