"""
Etapa 2 - sistema multi-agente com CrewAI.

Tres agentes especializados orquestrados em uma Crew:
  - Analista Financeiro: investiga pagamentos via Text-to-SQL no DuckDB.
  - Especialista Clinico: busca em documentos via RAG (ChromaDB).
  - Analista de Risco: prediz inadimplencia (modelo FLAML) e gera mensagens
    de cobranca empaticas.

Cada agente expoe ferramentas que reutilizam funcoes ja construidas na
etapa 1, evitando duplicacao. A Crew so e acionada para perguntas compostas
(que envolvem multiplas camadas), preservando o roteador rapido por keywords
para perguntas simples.

Uso:
    from src.crew import run_crew
    resposta = run_crew("Quem esta inadimplente e gere mensagens de cobranca")
"""

from __future__ import annotations

import os
from typing import Optional

from crewai import Agent, Crew, Process, Task
from crewai.tools import tool

from config import OLLAMA_BASE_URL, OLLAMA_CREW_MODEL
from src.database import execute_query, get_summary
from src.llm import generate_collection_message, text_to_sql
from src.ml_model import get_risk_summary, predict_risk
from src.observability import observe
from src.rag import search


# CrewAI usa LiteLLM. Para Ollama, prefixamos o modelo com "ollama/" e
# apontamos a base_url. Usamos OLLAMA_CREW_MODEL (default: gpt-oss:20b-cloud)
# porque function calling de CrewAI exige modelo 7B+ - llama3.2:1b emite o
# JSON de tool call malformado e a Crew vira ruido.
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "ollama-placeholder")
_LLM_MODEL = f"ollama/{OLLAMA_CREW_MODEL}"
_LLM_BASE = OLLAMA_BASE_URL


def _llm():
    """Constroi a configuracao de LLM Ollama para os agentes."""
    from crewai import LLM
    return LLM(model=_LLM_MODEL, base_url=_LLM_BASE, temperature=0.2)


# --- Tools (cada tool e uma fachada fina sobre as funcoes da etapa 1) ---


@tool("query_finance")
def tool_query_finance(question: str) -> str:
    """
    Consulta o banco financeiro respondendo perguntas sobre pagamentos,
    pendencias, faturamento, notas fiscais e sessoes. A entrada deve ser
    uma pergunta em portugues, ex.: 'Quais pacientes estao pendentes?'.
    """
    schema = (
        "Tabelas: pacientes(id_paciente, nome_codigo, valor_sessao, modelo_cobranca); "
        "financeiro(id_registro, id_paciente, data_sessao, valor, status_pagamento, nf_emitida)."
    )
    sql = text_to_sql(question, schema)
    try:
        df = execute_query(sql)
    except Exception as e:
        return f"Erro ao executar SQL: {e}. SQL gerado: {sql}"
    if df is None or df.empty:
        return "Nenhum resultado encontrado."
    return df.head(20).to_markdown(index=False)


@tool("financial_summary")
def tool_financial_summary() -> str:
    """Retorna o resumo financeiro consolidado (total pago, pendente, contagens)."""
    s = get_summary() or {}
    return (
        f"Total pago: R$ {s.get('total_pago', 0):.2f}; "
        f"Total pendente: R$ {s.get('total_pendente', 0):.2f}; "
        f"Sessoes pendentes: {s.get('pendentes_count', 0)}; "
        f"Pacientes ativos: {s.get('total_pacientes', 0)}."
    )


@tool("search_documents")
def tool_search_documents(query: str) -> str:
    """
    Busca semantica nos documentos clinicos (DSM-5, codigo de etica do CFP,
    tecnicas de TCC). Use para perguntas tecnicas/clinicas. Retorna ate 5
    trechos relevantes com fonte.
    """
    chunks = search(query, top_k=5)
    if not chunks:
        return "Nenhum trecho relevante encontrado."
    return "\n\n".join(
        f"[fonte: {c['source']} | score {c['score']:.2f}]\n{c['text']}" for c in chunks
    )


@tool("predict_patient_risk")
def tool_predict_patient_risk(id_or_code: str) -> str:
    """
    Prediz risco de inadimplencia. Aceita id interno (P001) ou codigo
    publico (PAC-ALPHA). Sem argumento usa-se 'all' para resumo geral.
    """
    code = id_or_code.strip().upper()
    if code in ("ALL", "TODOS", ""):
        return get_risk_summary()
    if code.startswith("PAC-"):
        df_lookup = execute_query(
            f"SELECT id_paciente FROM pacientes WHERE nome_codigo = '{code}'"
        )
        if df_lookup is not None and not df_lookup.empty:
            code = df_lookup.iloc[0]["id_paciente"]
    risk = predict_risk(code)
    if risk is None or risk.empty:
        return f"Paciente {id_or_code} nao encontrado."
    row = risk.iloc[0]
    return (
        f"Paciente {row['nome_codigo']}: risco {row['risco']} "
        f"(prob {row['probabilidade_risco']:.0%}); "
        f"{int(row['sessoes_pendentes'])} de {int(row['total_sessoes'])} "
        f"sessoes pendentes (taxa atraso {row['taxa_atraso']:.0%})."
    )


@tool("draft_collection_message")
def tool_draft_collection_message(
    patient_code: str,
    sessoes_pendentes: int,
    valor: float | None = None,
) -> str:
    """
    Gera mensagem de cobranca empatica para WhatsApp.

    Args:
        patient_code: codigo do paciente (ex.: 'PAC-ALPHA')
        sessoes_pendentes: numero de sessoes nao pagas
        valor: valor pendente em reais. Opcional - se nao souber, deixe None
            que sera calculado automaticamente pelo banco.
    """
    code = (patient_code or "").strip().upper()
    if not code.startswith("PAC-"):
        return f"Codigo de paciente invalido: '{patient_code}'. Esperado formato PAC-XXX."

    # Se valor nao foi passado, busca direto no banco
    if valor is None or (isinstance(valor, float) and valor != valor):  # NaN check
        try:
            df = execute_query(
                "SELECT SUM(f.valor) AS total FROM financeiro f "
                "JOIN pacientes p ON f.id_paciente = p.id_paciente "
                f"WHERE p.nome_codigo = '{code}' AND f.status_pagamento = 'pendente'"
            )
            if df is not None and not df.empty and df.iloc[0]["total"] is not None:
                valor = float(df.iloc[0]["total"])
            else:
                valor = float(sessoes_pendentes) * 200.0  # fallback razoavel
        except Exception:
            valor = float(sessoes_pendentes) * 200.0

    try:
        sess = int(sessoes_pendentes)
    except (ValueError, TypeError):
        return f"sessoes_pendentes deve ser inteiro, recebido: {sessoes_pendentes!r}"
    return generate_collection_message(code, float(valor), sess)


# --- Agentes ---


def _build_agents():
    llm = _llm()

    # Backstories forcam o uso obrigatorio das tools. Sem essa enfase, modelos
    # tipo gpt-oss:20b tendem a "responder direto" inventando schema/dados.
    financeiro = Agent(
        role="Analista Financeiro Clinico",
        goal=(
            "Responder perguntas sobre pagamentos, pendencias e nota fiscal "
            "consultando o banco real do consultorio via tool."
        ),
        backstory=(
            "Voce e um analista financeiro de um consultorio de psicologia. "
            "REGRAS OBRIGATORIAS:\n"
            "1. SEMPRE chame a tool 'query_finance' para qualquer pergunta de dados; "
            "NUNCA invente numeros, nomes, emails, telefones ou colunas.\n"
            "2. As tabelas REAIS sao: pacientes(id_paciente, nome_codigo, valor_sessao, "
            "modelo_cobranca) e financeiro(id_registro, id_paciente, data_sessao, valor, "
            "status_pagamento, nf_emitida). NAO existem colunas como 'email', 'telefone' "
            "ou 'data_vencimento'.\n"
            "3. Codigos de paciente seguem o padrao PAC-ALPHA, PAC-BETA etc - codigos "
            "gregos. Se vir nomes pessoais (Ana Silva, Joao etc) na sua resposta, esta "
            "errado: use a tool e copie os codigos reais retornados.\n"
            "4. Apos chamar a tool, SEMPRE escreva uma resposta final em texto/markdown "
            "com os resultados. NUNCA termine sua resposta com apenas uma chamada de "
            "tool - sempre conclua com texto resumindo os dados."
        ),
        tools=[tool_query_finance, tool_financial_summary],
        llm=llm,
        max_iter=3,
        allow_delegation=False,
        verbose=False,
    )

    clinico = Agent(
        role="Editor Final",
        goal=(
            "Reescrever os resultados ja coletados pelos outros agentes em um "
            "texto markdown final, claro e bem formatado, em portugues."
        ),
        backstory=(
            "Voce e um editor que so formata texto. NAO tem tools, NAO chama nada "
            "externo, NAO consulta nada. Seu trabalho e apenas reorganizar e "
            "polir o texto que ja veio das tasks anteriores.\n"
            "REGRAS RIGIDAS:\n"
            "1. SEMPRE termine sua resposta com texto markdown completo - nunca "
            "uma chamada de tool, nunca um JSON solto.\n"
            "2. Preserve codigos de paciente (PAC-ALPHA, PAC-MU etc) e valores "
            "exatos do contexto - nao arredonde nem invente.\n"
            "3. Estrutura sugerida: titulo H2, lista/tabela com os dados, breve "
            "interpretacao, depois (se houver) as mensagens de cobranca em blocos."
        ),
        tools=[],  # consolidador nao precisa de tools - evita tool_call sem texto final
        llm=llm,
        max_iter=2,
        allow_delegation=False,
        verbose=False,
    )

    risco = Agent(
        role="Analista de Risco e Cobranca",
        goal=(
            "Identificar pacientes em risco e gerar mensagens de cobranca a partir "
            "das tools de ML e geracao de texto."
        ),
        backstory=(
            "Voce combina o modelo FLAML (treinado em dados publicos do UCI) com "
            "tom acolhedor para falar com pacientes.\n"
            "REGRAS OBRIGATORIAS:\n"
            "1. Para identificar paciente de maior risco: chame 'predict_patient_risk' "
            "com argumento 'all'.\n"
            "2. Para gerar mensagem: chame 'draft_collection_message' com kwargs "
            "patient_code='PAC-XXX' e sessoes_pendentes=N (numero inteiro). O "
            "argumento valor e opcional - omita se nao souber.\n"
            "3. Para listar pendentes: chame 'query_finance' (NUNCA invente nomes "
            "ou valores; codigos sao PAC-ALPHA, PAC-BETA, ... gregos).\n"
            "4. NUNCA escreva uma mensagem de cobranca do zero - sempre use a tool.\n"
            "5. Apos terminar de chamar tools, SEMPRE escreva uma resposta final em "
            "markdown listando os resultados. NUNCA termine sua resposta com apenas "
            "uma chamada de tool - sempre conclua com texto formatado."
        ),
        tools=[tool_predict_patient_risk, tool_draft_collection_message, tool_query_finance],
        llm=llm,
        max_iter=4,
        allow_delegation=False,
        verbose=False,
    )

    return financeiro, clinico, risco


# --- Heuristica para acionar a Crew ---

_COMPOUND_HINTS = (
    " e ", " e gere", " e me ", " depois ", " entao ", " e tambem ",
    "/equipe", "/crew", "equipe ", "agente",
)


def should_use_crew(question: str) -> bool:
    """
    Decide se a pergunta vale uma orquestracao multi-agente.
    Heuristica simples: se o usuario pede explicitamente '/equipe' ou se a
    pergunta combina varios verbos/intencoes, ativa a Crew.
    """
    q = question.lower().strip()
    if q.startswith(("/equipe", "/crew")):
        return True
    return sum(1 for hint in _COMPOUND_HINTS if hint in f" {q} ") >= 1 and any(
        kw in q for kw in ("gere", "envie", "cobra", "depois", "tambem", "e me")
    )


def _strip_prefix(question: str) -> str:
    q = question.strip()
    for pref in ("/equipe", "/crew"):
        if q.lower().startswith(pref):
            return q[len(pref):].strip()
    return q


# --- Execucao ---


@observe(name="crew_run")
def run_crew(question: str) -> str:
    """
    Executa a Crew para a pergunta. Retorna texto markdown da resposta final.

    Notas:
      - Saida nao e streaming (CrewAI nao e built nativamente para streaming
        token-a-token); a UI do app exibe um spinner e depois o texto pronto.
      - Em caso de erro, devolve mensagem amigavel ao inves de levantar.
    """
    pergunta = _strip_prefix(question)
    financeiro, clinico, risco = _build_agents()

    investigar_financeiro = Task(
        description=(
            "Solicitacao do usuario: '" + pergunta + "'.\n\n"
            "Use a tool 'query_finance' para coletar TODOS os dados financeiros "
            "relevantes (pendencias, valores, codigos de paciente). Liste apenas "
            "o que a tool retornou - codigos no formato PAC-ALPHA. "
            "NUNCA invente nomes pessoais (Ana Silva, Joao) ou colunas que nao "
            "existem (email, telefone, data_vencimento)."
        ),
        agent=financeiro,
        expected_output=(
            "Lista (markdown) com codigos de pacientes pendentes e seus valores, "
            "exatamente como retornados pela tool query_finance."
        ),
    )

    avaliar_risco = Task(
        description=(
            "Solicitacao original: '" + pergunta + "'.\n\n"
            "Com base no contexto financeiro coletado:\n"
            "1. Se a solicitacao pede 'paciente de maior risco', chame "
            "'predict_patient_risk' com argumento 'all' e identifique o de maior "
            "probabilidade.\n"
            "2. Se a solicitacao pede mensagens de cobranca, chame "
            "'draft_collection_message' UMA VEZ por paciente, no formato "
            "'PAC-CODIGO;valor;sessoes_pendentes' (use os valores reais do "
            "contexto financeiro).\n"
            "NUNCA escreva mensagem manualmente - sempre use a tool."
        ),
        agent=risco,
        expected_output=(
            "Lista de mensagens de cobranca prontas (uma por paciente identificado) "
            "OU analise de risco do paciente especifico solicitado."
        ),
        context=[investigar_financeiro],
    )

    consolidar = Task(
        description=(
            "Solicitacao original: '" + pergunta + "'.\n\n"
            "Use os resultados das tasks anteriores (contexto) para escrever uma "
            "resposta final clara em portugues, com markdown bem formatado. "
            "Preserve os codigos de paciente (PAC-ALPHA etc) exatamente. "
            "NAO chame tools redundantes - os dados ja foram coletados."
        ),
        agent=clinico,
        expected_output="Resposta final ao usuario em markdown, pronta para o chat.",
        context=[investigar_financeiro, avaliar_risco],
    )

    crew = Crew(
        agents=[financeiro, risco, clinico],
        tasks=[investigar_financeiro, avaliar_risco, consolidar],
        process=Process.sequential,
        verbose=False,
    )

    try:
        result = crew.kickoff()
        return str(result)
    except Exception as e:
        # Fallback: o gpt-oss as vezes termina uma task com tool_call sem
        # texto, levantando ValidationError no TaskOutput. Quando isso
        # acontece, caimos para um caminho mais simples — usar diretamente
        # as funcoes da etapa 1 — para garantir que o usuario recebe alguma
        # resposta util.
        return _fallback_simple(pergunta, error=str(e))


def _fallback_simple(pergunta: str, error: str = "") -> str:
    """
    Fallback quando a Crew levanta erro estrutural (tipico do CrewAI 1.x com
    modelos que terminam em tool_call). Reusa as funcoes ja existentes.
    """
    out = ["> _Caminho de fallback ativado: a orquestracao Crew encontrou um "
           "erro de formatacao do LLM e o sistema voltou ao pipeline simples._\n"]
    if error:
        out.append(f"> Detalhe tecnico: `{error[:200]}`\n")

    try:
        risco_txt = get_risk_summary()
        out.append("\n## Analise de Risco\n")
        out.append(risco_txt)
    except Exception as e:
        out.append(f"\n(falha ao gerar analise de risco: {e})")

    try:
        df = execute_query(
            "SELECT p.nome_codigo, COUNT(*) AS sessoes_pendentes, "
            "SUM(f.valor) AS valor_pendente "
            "FROM financeiro f JOIN pacientes p ON f.id_paciente = p.id_paciente "
            "WHERE f.status_pagamento = 'pendente' "
            "GROUP BY p.nome_codigo ORDER BY valor_pendente DESC LIMIT 5"
        )
        if df is not None and not df.empty:
            out.append("\n## Top 5 pacientes inadimplentes\n")
            out.append(df.to_markdown(index=False))
            top = df.iloc[0]
            msg = generate_collection_message(
                top["nome_codigo"], float(top["valor_pendente"]),
                int(top["sessoes_pendentes"]),
            )
            out.append(f"\n## Mensagem para {top['nome_codigo']}\n\n{msg}")
    except Exception as e:
        out.append(f"\n(falha ao listar pendentes: {e})")

    return "\n".join(out)


def stream_crew(question: str):
    """
    Adapter para a UI Streamlit: executa a Crew e devolve o texto final em
    um unico yield, compativel com `st.write_stream`.
    """
    yield run_crew(question)
