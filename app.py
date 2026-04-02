"""
Interface Streamlit - Assistente de Gestao para Psicologa Clinica.
Ponto de entrada: streamlit run app.py
"""

import streamlit as st
from config import MSG_BEM_VINDA, MSG_CONSULTANDO_FINANCEIRO, MSG_BUSCANDO_DOCS, MSG_ANALISANDO_RISCO, MSG_ERRO_LLM
from src.llm import check_ollama
from src.database import load_tables, get_summary
from src.rag import index_documents, get_stats
from src.ml_model import train_model
from src.agents import process_question, route_question

# --- Configuracao da Pagina ---
st.set_page_config(
    page_title="Assistente Clinica",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded",
)

# --- CSS Customizado ---
st.markdown("""
<style>
    .stChatMessage {
        padding: 0.5rem 1rem;
    }
    .status-ok { color: #28a745; font-weight: bold; }
    .status-err { color: #dc3545; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


# --- Inicializacao (executada uma vez) ---
@st.cache_resource
def initialize_system():
    """Inicializa todos os componentes do sistema."""
    status = {"ollama": False, "database": None, "rag": None, "ml": None, "errors": []}

    # 1. Verificar Ollama
    status["ollama"] = check_ollama()
    if not status["ollama"]:
        status["errors"].append("Ollama nao esta rodando")

    # 2. Carregar banco de dados
    try:
        db_info = load_tables()
        status["database"] = db_info
    except Exception as e:
        status["errors"].append(f"Erro ao carregar dados: {str(e)}")

    # 3. Indexar documentos (se houver)
    try:
        rag_info = index_documents()
        status["rag"] = rag_info
    except Exception as e:
        status["errors"].append(f"Erro ao indexar documentos: {str(e)}")

    # 4. Treinar modelo ML
    try:
        if status["database"]:
            ml_info = train_model()
            status["ml"] = ml_info
    except Exception as e:
        status["errors"].append(f"Erro ao treinar modelo: {str(e)}")

    return status


# --- Sidebar ---
def render_sidebar(status):
    """Renderiza a sidebar com informacoes do sistema."""
    with st.sidebar:
        st.title("🧠 Assistente Clinica")
        st.markdown("---")

        # Status do sistema
        st.subheader("Status do Sistema")

        # Ollama
        if status["ollama"]:
            st.markdown("✅ **Ollama**: Conectado")
        else:
            st.markdown("❌ **Ollama**: Desconectado")
            st.warning("Inicie o Ollama com `ollama serve`")

        # Banco de dados
        if status["database"]:
            db = status["database"]
            st.markdown(f"✅ **Dados**: {db['pacientes']} pacientes, {db['financeiro']} registros")
        else:
            st.markdown("❌ **Dados**: Nao carregados")

        # RAG
        if status["rag"] and status["rag"].get("status") == "ok":
            rag = status["rag"]
            st.markdown(f"✅ **Documentos**: {rag['total_files']} arquivo(s), {rag['total_chunks']} chunks")
        else:
            st.markdown("📄 **Documentos**: Nenhum indexado")
            st.info("Coloque arquivos em `docs/`")

        # Modelo ML
        if status["ml"]:
            ml = status["ml"]
            st.markdown(f"✅ **Modelo ML**: Acuracia {ml['accuracy']:.0%}")
            st.markdown(f"   - Risco alto: {ml['risco_alto']} pacientes")
            st.markdown(f"   - Risco baixo: {ml['risco_baixo']} pacientes")
        else:
            st.markdown("⚠️ **Modelo ML**: Nao treinado")

        st.markdown("---")

        # Resumo financeiro
        if status["database"]:
            summary = get_summary()
            if summary:
                st.subheader("Resumo Financeiro")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Pago", f"R$ {summary['total_pago']:,.2f}")
                with col2:
                    st.metric("Pendente", f"R$ {summary['total_pendente']:,.2f}")
                st.metric("Sessoes Pendentes", summary["pendentes_count"])

        st.markdown("---")
        st.caption("Assistente de Gestao Clinica v1.0")
        st.caption("Powered by Ollama + DuckDB + ChromaDB")


# --- Chat Interface ---
def render_chat(status):
    """Renderiza a interface de chat principal."""
    st.title("Assistente de Gestao Clinica")

    # Inicializar historico
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Mostrar mensagem de boas-vindas
    if not st.session_state.messages:
        with st.chat_message("assistant", avatar="🧠"):
            st.markdown(MSG_BEM_VINDA)

    # Mostrar historico
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🧠" if msg["role"] == "assistant" else "👤"):
            st.markdown(msg["content"])

    # Input do usuario
    if prompt := st.chat_input("Digite sua pergunta..."):
        # Adicionar mensagem do usuario
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        # Verificar se Ollama esta rodando
        if not status["ollama"]:
            with st.chat_message("assistant", avatar="🧠"):
                st.error(MSG_ERRO_LLM)
            return

        # Detectar tipo de agente e mostrar feedback
        agent_type = route_question(prompt)
        feedback_messages = {
            "financeiro": MSG_CONSULTANDO_FINANCEIRO,
            "rag": MSG_BUSCANDO_DOCS,
            "risco": MSG_ANALISANDO_RISCO,
            "cobranca": MSG_CONSULTANDO_FINANCEIRO,
            "geral": None,
        }

        with st.chat_message("assistant", avatar="🧠"):
            feedback = feedback_messages.get(agent_type)
            if feedback:
                with st.status(feedback, expanded=False):
                    st.write(f"Agente: {agent_type}")

            # Processar e exibir resposta com streaming
            try:
                _, response_generator = process_question(prompt)
                response = st.write_stream(response_generator)
            except Exception as e:
                response = f"Ocorreu um erro ao processar sua pergunta: {str(e)}"
                st.error(response)

        # Salvar resposta no historico
        st.session_state.messages.append({"role": "assistant", "content": response})


# --- Main ---
def main():
    status = initialize_system()
    render_sidebar(status)
    render_chat(status)


if __name__ == "__main__":
    main()
