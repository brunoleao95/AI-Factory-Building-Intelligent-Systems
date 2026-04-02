"""
Comunicacao com Ollama via API REST.
Funcoes: chat com streaming, Text-to-SQL, RAG answer, mensagem de cobranca.
"""

import json
import requests
from config import OLLAMA_BASE_URL, OLLAMA_MODEL, SYSTEM_PROMPT


def check_ollama():
    """Verifica se o Ollama esta rodando."""
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        return r.status_code == 200
    except requests.ConnectionError:
        return False


def chat_stream(messages, system_prompt=None):
    """
    Envia mensagens para o Ollama e retorna um generator de tokens (streaming).

    Args:
        messages: Lista de dicts [{"role": "user"/"assistant", "content": "..."}]
        system_prompt: System prompt customizado (opcional, usa padrao se None)

    Yields:
        Tokens da resposta (strings)
    """
    if system_prompt is None:
        system_prompt = SYSTEM_PROMPT

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [{"role": "system", "content": system_prompt}] + messages,
        "stream": True,
    }

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json=payload,
            stream=True,
            timeout=120,
        )
        response.raise_for_status()

        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                if "message" in data and "content" in data["message"]:
                    yield data["message"]["content"]
                if data.get("done", False):
                    break

    except requests.ConnectionError:
        yield "Erro: Ollama nao esta rodando. Inicie com `ollama serve`."
    except requests.Timeout:
        yield "Erro: Timeout na conexao com o Ollama."
    except Exception as e:
        yield f"Erro inesperado: {str(e)}"


def chat(messages, system_prompt=None):
    """
    Versao nao-streaming do chat. Retorna resposta completa como string.
    """
    tokens = list(chat_stream(messages, system_prompt))
    return "".join(tokens)


def text_to_sql(question, schema_description):
    """
    Converte pergunta em linguagem natural para SQL usando o LLM.

    Args:
        question: Pergunta do usuario
        schema_description: Descricao das tabelas do DuckDB

    Returns:
        String com a query SQL
    """
    prompt = f"""Voce e um especialista em SQL. Converta a pergunta do usuario em uma query SQL valida.

SCHEMA DO BANCO DE DADOS:
{schema_description}

REGRAS:
- Use apenas as tabelas e colunas descritas acima
- Retorne APENAS a query SQL, sem explicacoes
- Use aspas simples para strings
- Formate valores monetarios com 2 casas decimais
- Para datas, use formato 'YYYY-MM-DD'
- Use aliases claros para colunas calculadas
- Nunca use DELETE, UPDATE, INSERT, DROP ou ALTER

PERGUNTA: {question}

SQL:"""

    messages = [{"role": "user", "content": prompt}]
    response = chat(messages, system_prompt="Voce converte perguntas em SQL. Retorne APENAS o SQL, sem markdown, sem explicacao.")

    # Limpar resposta (remover markdown code blocks se houver)
    sql = response.strip()
    sql = sql.replace("```sql", "").replace("```", "").strip()

    return sql


def rag_answer(question, context_chunks):
    """
    Gera resposta usando contexto de documentos recuperados (RAG).

    Args:
        question: Pergunta do usuario
        context_chunks: Lista de strings com trechos relevantes dos documentos

    Returns:
        Generator de tokens (streaming)
    """
    context = "\n\n---\n\n".join(context_chunks)

    prompt = f"""Responda a pergunta do usuario com base APENAS nos trechos de documentos fornecidos abaixo.

DOCUMENTOS RELEVANTES:
{context}

REGRAS:
- Baseie sua resposta APENAS nas informacoes dos documentos acima
- Cite a fonte quando possivel (nome do documento)
- Se os documentos nao contiverem a resposta, diga claramente
- Use linguagem profissional e acolhedora
- Formate a resposta com markdown quando apropriado

PERGUNTA: {question}"""

    messages = [{"role": "user", "content": prompt}]
    return chat_stream(messages, system_prompt=SYSTEM_PROMPT)


def generate_collection_message(patient_code, valor, sessoes_pendentes):
    """
    Gera mensagem de cobranca respeitosa para WhatsApp.

    Args:
        patient_code: Codigo do paciente (ex: PAC-ALPHA)
        valor: Valor total pendente
        sessoes_pendentes: Numero de sessoes pendentes

    Returns:
        String com a mensagem formatada
    """
    prompt = f"""Gere uma mensagem de cobranca respeitosa e empatica para enviar via WhatsApp.

DADOS:
- Paciente: {patient_code}
- Valor pendente: R$ {valor:.2f}
- Sessoes pendentes: {sessoes_pendentes}

REGRAS:
- Tom respeitoso e empatico (lembre-se que e uma relacao terapeutica)
- Nao use o codigo do paciente na mensagem (use "voce")
- Ofereca opcoes de parcelamento ou negociacao
- Seja breve (maximo 4-5 linhas)
- Nao use emojis excessivos (maximo 1-2)
- Inclua saudacao e despedida

MENSAGEM:"""

    messages = [{"role": "user", "content": prompt}]
    return chat(messages, system_prompt="Voce gera mensagens de cobranca profissionais e empaticas para uma psicologa clinica.")
