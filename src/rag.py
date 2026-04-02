"""
Pipeline RAG: indexacao de documentos e busca semantica com ChromaDB.
"""

import os
import hashlib
import chromadb
from chromadb.utils import embedding_functions
from config import DOCS_DIR, CHROMA_DIR, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, RAG_TOP_K


def _get_embedding_function():
    """Retorna funcao de embedding do sentence-transformers."""
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )


def _get_collection():
    """Retorna colecao do ChromaDB (cria se nao existir)."""
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    ef = _get_embedding_function()
    collection = client.get_or_create_collection(
        name="documentos",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def _read_file(filepath):
    """Le conteudo de um arquivo (TXT ou PDF)."""
    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".txt":
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()

    elif ext == ".pdf":
        try:
            from pypdf import PdfReader
            reader = PdfReader(filepath)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text
        except ImportError:
            print(f"  AVISO: pypdf nao instalado. Ignorando {filepath}")
            return ""

    return ""


def _chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Divide texto em chunks com overlap.

    Args:
        text: Texto completo
        chunk_size: Tamanho maximo de cada chunk (caracteres)
        overlap: Sobreposicao entre chunks (caracteres)

    Returns:
        Lista de strings (chunks)
    """
    if not text or len(text.strip()) == 0:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size

        # Tentar quebrar em espaco ou ponto para nao cortar palavras
        if end < len(text):
            # Procurar ultimo espaco ou ponto antes do limite
            for sep in [". ", "\n", " "]:
                last_sep = text.rfind(sep, start, end)
                if last_sep > start:
                    end = last_sep + len(sep)
                    break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - overlap
        if start >= len(text):
            break

    return chunks


def index_documents(docs_path=None):
    """
    Indexa todos os documentos da pasta docs/ no ChromaDB.

    Args:
        docs_path: Caminho para pasta de documentos (usa padrao se None)

    Returns:
        Dict com estatisticas da indexacao
    """
    if docs_path is None:
        docs_path = DOCS_DIR

    if not os.path.exists(docs_path):
        return {"status": "error", "message": f"Pasta nao encontrada: {docs_path}"}

    # Listar arquivos validos
    valid_extensions = {".txt", ".pdf"}
    files = [
        f for f in os.listdir(docs_path)
        if os.path.splitext(f)[1].lower() in valid_extensions
    ]

    if not files:
        return {"status": "empty", "message": "Nenhum documento encontrado na pasta docs/"}

    collection = _get_collection()

    total_chunks = 0
    indexed_files = []

    for filename in files:
        filepath = os.path.join(docs_path, filename)
        text = _read_file(filepath)

        if not text.strip():
            continue

        chunks = _chunk_text(text)

        if not chunks:
            continue

        # Gerar IDs unicos baseados no conteudo
        ids = []
        metadatas = []
        for i, chunk in enumerate(chunks):
            chunk_id = hashlib.md5(f"{filename}_{i}_{chunk[:50]}".encode()).hexdigest()
            ids.append(chunk_id)
            metadatas.append({
                "source": filename,
                "chunk_index": i,
                "total_chunks": len(chunks),
            })

        # Upsert no ChromaDB (adiciona ou atualiza)
        collection.upsert(
            ids=ids,
            documents=chunks,
            metadatas=metadatas,
        )

        total_chunks += len(chunks)
        indexed_files.append(filename)

    return {
        "status": "ok",
        "files": indexed_files,
        "total_chunks": total_chunks,
        "total_files": len(indexed_files),
    }


def search(query, top_k=None):
    """
    Busca semantica nos documentos indexados.

    Args:
        query: Pergunta ou termo de busca
        top_k: Numero de resultados (usa padrao se None)

    Returns:
        Lista de dicts com 'text', 'source', 'score'
    """
    if top_k is None:
        top_k = RAG_TOP_K

    collection = _get_collection()

    if collection.count() == 0:
        return []

    results = collection.query(
        query_texts=[query],
        n_results=min(top_k, collection.count()),
    )

    output = []
    if results and results["documents"]:
        for i, doc in enumerate(results["documents"][0]):
            output.append({
                "text": doc,
                "source": results["metadatas"][0][i].get("source", "desconhecido"),
                "score": 1 - results["distances"][0][i] if results["distances"] else None,
            })

    return output


def is_indexed():
    """Verifica se ha documentos indexados no ChromaDB."""
    try:
        collection = _get_collection()
        return collection.count() > 0
    except Exception:
        return False


def get_stats():
    """Retorna estatisticas da colecao."""
    try:
        collection = _get_collection()
        count = collection.count()
        return {"indexed": count > 0, "total_chunks": count}
    except Exception:
        return {"indexed": False, "total_chunks": 0}
