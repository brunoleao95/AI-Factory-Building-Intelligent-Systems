"""
Configuracoes, constantes e system prompt do assistente.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# --- Ollama ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
# Modelo dedicado para agentes CrewAI. Function calling exige modelo robusto
# (>= 7B parametros). Default cai no OLLAMA_MODEL para nao quebrar setups antigos.
OLLAMA_CREW_MODEL = os.getenv("OLLAMA_CREW_MODEL", OLLAMA_MODEL)

# --- Caminhos ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DOCS_DIR = os.path.join(BASE_DIR, "docs")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_data")

# --- ChromaDB / RAG ---
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
RAG_TOP_K = 5

# --- System Prompt ---
SYSTEM_PROMPT = """Voce e uma assistente de gestao para uma psicologa clinica autonoma.
Seu papel e ajudar com:
1. Consultas financeiras (pagamentos, inadimplencia, notas fiscais)
2. Pesquisa em documentos tecnicos (DSM-5, tecnicas TCC, etica profissional)
3. Geracao de mensagens de cobranca respeitosas para WhatsApp
4. Analise de risco de inadimplencia de pacientes

Regras:
- Use linguagem profissional mas acolhedora
- Nunca revele nomes reais de pacientes (use apenas os codigos como PAC-ALPHA)
- Ao consultar dados financeiros, sempre formate valores em R$
- Ao responder sobre documentos, cite a fonte (nome do arquivo e trecho)
- Se nao souber a resposta, diga claramente
- Formate respostas com markdown quando apropriado (tabelas, listas, negrito)
- Seja concisa e objetiva, mas sem perder o tom acolhedor"""

# --- Mensagens da Interface ---
MSG_BEM_VINDA = (
    "Ola! Sou sua assistente de gestao clinica. Posso ajudar com:\n\n"
    "- **Consultas financeiras** (pagamentos, inadimplencia, notas fiscais)\n"
    "- **Pesquisa em documentos** tecnicos (DSM-5, TCC, etica)\n"
    "- **Analise de risco** de inadimplencia de pacientes\n"
    "- **Mensagens de cobranca** respeitosas para WhatsApp\n"
    "- **Equipe de agentes** (CrewAI): para tarefas compostas, comece com `/equipe` "
    "ou faca varias perguntas em uma so. Ex.: `/equipe quem esta inadimplente e gere "
    "as mensagens de cobranca`.\n\n"
    "Como posso ajudar?"
)

MSG_CONSULTANDO_FINANCEIRO = "Consultando seus dados financeiros..."
MSG_BUSCANDO_DOCS = "Buscando nos documentos tecnicos..."
MSG_ANALISANDO_RISCO = "Analisando risco de inadimplencia..."
MSG_EQUIPE_TRABALHANDO = "Equipe de agentes coordenando a resposta..."
MSG_ERRO_LLM = "Desculpe, nao consegui me conectar ao modelo de linguagem. Verifique se o Ollama esta rodando."
MSG_ERRO_QUERY = "Nao consegui processar essa consulta. Tente reformular a pergunta."
MSG_SEM_DOCS = "Nenhum documento foi indexado ainda. Coloque arquivos na pasta docs/ e reinicie o app."
MSG_SEM_RESULTADO = "Nao encontrei informacoes sobre isso nos seus documentos. Tente reformular a pergunta."
