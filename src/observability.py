"""
Etapa 2 - observabilidade com Langfuse.

Instrumenta o pipeline pergunta -> retrieval -> LLM -> resposta. Se as chaves
do Langfuse nao estiverem definidas no .env, exporta um decorator no-op para
nao quebrar o app local.

Uso:
    from src.observability import observe, langfuse_client

    @observe(name="text_to_sql")
    def text_to_sql(...): ...

Saida:
    Traces no dashboard cloud.langfuse.com com latencia, input, output e
    metadados (modelo, top_k de RAG, etc).
"""

from __future__ import annotations

import os
from functools import wraps
from typing import Any, Callable

from dotenv import load_dotenv

load_dotenv()

_PUBLIC = os.getenv("LANGFUSE_PUBLIC_KEY", "").strip().strip('"')
_SECRET = os.getenv("LANGFUSE_SECRET_KEY", "").strip().strip('"')
# Aceita LANGFUSE_HOST (nomenclatura nova) ou LANGFUSE_BASE_URL (legacy);
# default fallback para o endpoint EU.
_HOST = (
    os.getenv("LANGFUSE_HOST")
    or os.getenv("LANGFUSE_BASE_URL")
    or "https://cloud.langfuse.com"
).strip().strip('"')

LANGFUSE_ENABLED = bool(_PUBLIC and _SECRET)

_langfuse = None
_observe_decorator: Callable[..., Any] | None = None

if LANGFUSE_ENABLED:
    try:
        from langfuse import Langfuse, observe as _lf_observe

        _langfuse = Langfuse(
            public_key=_PUBLIC,
            secret_key=_SECRET,
            host=_HOST,
        )
        _observe_decorator = _lf_observe
    except Exception as e:  # pragma: no cover
        print(f"[observability] erro ao inicializar Langfuse: {e}. Instrumentacao desativada.")
        LANGFUSE_ENABLED = False


def observe(name: str | None = None, as_type: str | None = None):
    """
    Decorator unico do projeto. Quando Langfuse esta ativo delega ao
    `langfuse.observe`; caso contrario retorna a funcao original.
    Generators sao retornados intactos para nao consumir o stream do chat.
    """
    def decorator(func: Callable):
        if LANGFUSE_ENABLED and _observe_decorator is not None:
            kwargs: dict[str, Any] = {}
            if name:
                kwargs["name"] = name
            if as_type:
                kwargs["as_type"] = as_type
            return _observe_decorator(**kwargs)(func)

        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper

    return decorator


def update_trace_metadata(**metadata):
    """Atualiza metadata do trace atual (modelo, tokens, etc) se Langfuse ativo."""
    if not LANGFUSE_ENABLED or _langfuse is None:
        return
    try:
        _langfuse.update_current_trace(metadata=metadata)
    except Exception:
        pass


def flush():
    """Forca envio dos eventos pendentes ao servidor Langfuse."""
    if LANGFUSE_ENABLED and _langfuse is not None:
        try:
            _langfuse.flush()
        except Exception:
            pass


def status() -> dict:
    """Estado para a sidebar do app."""
    return {
        "enabled": LANGFUSE_ENABLED,
        "host": _HOST if LANGFUSE_ENABLED else None,
    }
