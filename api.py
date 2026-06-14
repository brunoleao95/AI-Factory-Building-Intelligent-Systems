"""
Etapa 3 - Backend FastAPI (local).

Expoe o assistente como API para o frontend publicado no Streamlit Cloud.
Roda na maquina do usuario (onde vivem Ollama + deps pesadas + guardrails) e e
exposto por ngrok (dominio estatico). NAO altera a Etapa 2: apenas envolve
`process_question` com a camada de seguranca (scan_input / scan_output).

Fluxo /chat:
  scan_input -> (se permitido) process_question (Etapa 2) -> scan_output ->
  texto sanitizado re-transmitido em chunks via SSE (PII nunca vaza no stream).

Subir:  uvicorn api:app --host 0.0.0.0 --port 8000
"""

import json
import os

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

load_dotenv()

from src.agents import process_question, route_question  # noqa: E402
from src.crew import should_use_crew  # noqa: E402
from src.database import get_summary, load_tables  # noqa: E402
from src.guardrails import scan_input, scan_output, warmup  # noqa: E402
from src.llm import check_ollama  # noqa: E402
from src.ml_model import train_model  # noqa: E402
from src.observability import flush as observability_flush  # noqa: E402
from src.rag import index_documents  # noqa: E402

API_KEY = os.getenv("API_KEY", "")

app = FastAPI(title="Assistente Clinica - Backend (Etapa 3)")

# O frontend roda em outro dominio (Streamlit Cloud) -> liberar CORS.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_STATE = {"ollama": False, "database": None, "rag": None, "ml": None, "errors": []}


@app.on_event("startup")
def _startup():
    """Inicializa os mesmos componentes do app.py + aquece os guardrails."""
    _STATE["ollama"] = check_ollama()
    try:
        _STATE["database"] = load_tables()
    except Exception as e:
        _STATE["errors"].append(f"db: {e}")
    try:
        _STATE["rag"] = index_documents()
    except Exception as e:
        _STATE["errors"].append(f"rag: {e}")
    try:
        if _STATE["database"]:
            _STATE["ml"] = train_model()
    except Exception as e:
        _STATE["errors"].append(f"ml: {e}")
    try:
        warmup()  # pre-carrega Presidio + modelo HF (reduz cold start)
    except Exception as e:
        _STATE["errors"].append(f"guardrails: {e}")


def _check_key(x_api_key):
    """Valida o segredo compartilhado. Se API_KEY estiver vazio, libera (dev)."""
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="API key invalida.")


def _sse(payload):
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


class ChatRequest(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/status")
def status(x_api_key: str = Header(None)):
    _check_key(x_api_key)
    db = _STATE["database"] or {}
    ml = _STATE["ml"] or {}
    rag = _STATE["rag"] or {}
    summary = get_summary() if _STATE["database"] else {}
    return {
        "ollama": _STATE["ollama"],
        "pacientes": db.get("pacientes"),
        "registros": db.get("financeiro"),
        "rag_chunks": rag.get("total_chunks"),
        "ml_estimator": ml.get("best_estimator"),
        "ml_roc_auc": ml.get("roc_auc"),
        "resumo_financeiro": summary,
        "guardrails": True,
        "errors": _STATE["errors"],
    }


@app.post("/chat")
def chat(req: ChatRequest, x_api_key: str = Header(None)):
    _check_key(x_api_key)
    question = (req.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Pergunta vazia.")

    # --- Guardrail de ENTRADA ---
    gin = scan_input(question)
    if not gin.allowed:
        def blocked_stream():
            yield _sse({"type": "meta", "agent": "guardrail", "blocked": True,
                        "category": gin.category})
            yield _sse({"type": "token", "text": gin.reason})
            yield _sse({"type": "done"})
        return StreamingResponse(blocked_stream(), media_type="text/event-stream")

    safe_question = gin.sanitized_text or question
    agent_type = "equipe" if should_use_crew(safe_question) else route_question(safe_question)

    def event_stream():
        yield _sse({"type": "meta", "agent": agent_type, "blocked": False})
        try:
            _, generator = process_question(safe_question)
            full = "".join(generator)
        except Exception as e:
            yield _sse({"type": "token", "text": f"Ocorreu um erro: {e}"})
            yield _sse({"type": "done"})
            return

        # --- Guardrail de SAIDA (sanitiza antes de exibir) ---
        gout = scan_output(full)
        safe_text = gout.sanitized_text or full

        # Re-transmite o texto sanitizado em chunks (UX de streaming, sem vazar PII)
        for chunk in _chunks(safe_text):
            yield _sse({"type": "token", "text": chunk})
        yield _sse({"type": "done"})
        observability_flush()

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def _chunks(text, size=24):
    """Quebra o texto em pedacos para exibicao progressiva no frontend."""
    for i in range(0, len(text), size):
        yield text[i:i + size]
