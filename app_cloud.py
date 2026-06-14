"""
Etapa 3 - Frontend leve (publicado no Streamlit Cloud).

NAO importa src/ nem config (que puxariam torch/chromadb/crewai). Depende
apenas de streamlit + requests + python-dotenv, conversando com o backend
FastAPI local (api.py) exposto por ngrok. Ponto de entrada: streamlit run app_cloud.py

Secrets (painel do Streamlit Cloud, ou .env local):
  BACKEND_URL  - URL publica do backend (ex.: https://seu-nome.ngrok-free.app)
  API_KEY      - segredo compartilhado, enviado no header X-API-Key
"""

import json
import os

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()


def _secret(key, default=""):
    """st.secrets (Streamlit Cloud) tem prioridade; cai em variavel de ambiente."""
    try:
        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return os.getenv(key, default)


BACKEND_URL = _secret("BACKEND_URL", "http://localhost:8000").rstrip("/")
API_KEY = _secret("API_KEY", "")
# ngrok-skip-browser-warning: o free tier do ngrok injeta uma pagina de aviso
# para requisicoes; o header pula isso e devolve o JSON/SSE direto.
HEADERS = {"ngrok-skip-browser-warning": "true"}
if API_KEY:
    HEADERS["X-API-Key"] = API_KEY

MSG_BEM_VINDA = (
    "Ola! Sou sua assistente de gestao clinica. Posso ajudar com:\n\n"
    "- **Consultas financeiras** (pagamentos, inadimplencia, notas fiscais)\n"
    "- **Pesquisa em documentos** tecnicos (DSM-5, TCC, etica)\n"
    "- **Analise de risco** de inadimplencia de pacientes\n"
    "- **Mensagens de cobranca** respeitosas para WhatsApp\n"
    "- **Equipe de agentes** (CrewAI): comece com `/equipe` para tarefas compostas.\n\n"
    "Como posso ajudar?"
)
MSG_DISCLAIMER_IA = (
    "Voce esta interagindo com uma **IA**. As respostas podem conter erros e nao "
    "substituem o julgamento profissional. Nao use nomes reais de pacientes."
)

st.set_page_config(page_title="Assistente Clinica", page_icon="🧠",
                   layout="centered", initial_sidebar_state="expanded")


def backend_status():
    try:
        r = requests.get(f"{BACKEND_URL}/status", headers=HEADERS, timeout=8)
        if r.status_code == 200:
            return r.json()
    except Exception:
        return None
    return None


def stream_answer(question):
    """Consome o SSE do backend e gera tokens para st.write_stream."""
    try:
        resp = requests.post(f"{BACKEND_URL}/chat", json={"question": question},
                             headers=HEADERS, stream=True, timeout=300)
        resp.raise_for_status()
    except requests.exceptions.RequestException:
        yield ("Nao consegui falar com o assistente. Verifique se o backend local "
               "esta ligado (uvicorn) e exposto pelo ngrok, e se a URL/API key estao corretas.")
        return

    for line in resp.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data:"):
            continue
        try:
            evt = json.loads(line[5:].strip())
        except json.JSONDecodeError:
            continue
        if evt.get("type") == "token":
            yield evt.get("text", "")
        elif evt.get("type") == "done":
            break


# --- Sidebar ---
with st.sidebar:
    st.title("🧠 Assistente Clinica")
    st.caption("Etapa 3 - frontend (Streamlit Cloud) + backend local")
    st.markdown("---")
    status = backend_status()
    if status:
        st.markdown("✅ **Backend**: conectado")
        st.markdown(f"{'✅' if status.get('ollama') else '❌'} **Ollama**: "
                    f"{'ativo' if status.get('ollama') else 'inativo'}")
        if status.get("pacientes"):
            st.markdown(f"✅ **Dados**: {status['pacientes']} pacientes, "
                        f"{status.get('registros')} registros")
        if status.get("ml_roc_auc") is not None:
            st.markdown(f"✅ **Modelo ML** ({status.get('ml_estimator')}): "
                        f"ROC-AUC {status['ml_roc_auc']:.2f}")
        st.markdown("🛡️ **Guardrails**: ativos (LLM Guard-equivalente + Presidio)")
        resumo = status.get("resumo_financeiro") or {}
        if resumo:
            st.markdown("---")
            st.subheader("Resumo Financeiro")
            c1, c2 = st.columns(2)
            c1.metric("Total Pago", f"R$ {resumo.get('total_pago', 0):,.2f}")
            c2.metric("Pendente", f"R$ {resumo.get('total_pendente', 0):,.2f}")
    else:
        st.markdown("❌ **Backend**: offline")
        st.warning("O assistente precisa estar ligado na maquina da psicologa "
                   "(uvicorn + ngrok). Configure BACKEND_URL e API_KEY nos secrets.")
    st.markdown("---")
    st.caption("Assistente de Gestao Clinica v3.0")


# --- Chat ---
st.title("Assistente de Gestao Clinica")
st.info(MSG_DISCLAIMER_IA, icon="⚠️")

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    with st.chat_message("assistant", avatar="🧠"):
        st.markdown(MSG_BEM_VINDA)

for msg in st.session_state.messages:
    avatar = "🧠" if msg["role"] == "assistant" else "👤"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

if prompt := st.chat_input("Digite sua pergunta..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)
    with st.chat_message("assistant", avatar="🧠"):
        with st.status("Consultando o assistente...", expanded=False):
            st.write(f"Backend: {BACKEND_URL}")
        response = st.write_stream(stream_answer(prompt))
    st.session_state.messages.append({"role": "assistant", "content": response})
