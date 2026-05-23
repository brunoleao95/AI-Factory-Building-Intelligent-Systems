# Etapa 1 - Construcao da Base

> Relatorio retrospectivo. Documenta o estado do projeto entregue na etapa 1 (CBL: engajar). Serve como ponto de referencia para garantir que a etapa 2 nao quebre nada.

## Engajar (CBL)

- **Grande ideia**: gestao inteligente e produtividade tecnologica na psicologia clinica.
- **Pergunta essencial**: como uma psicologa autonoma pode centralizar gestao financeira e dominio tecnico em uma unica interface conversacional?
- **Desafio**: construir um assistente que (a) responde perguntas sobre o consultorio em linguagem natural, (b) consulta material tecnico (DSM-5, codigo de etica, tecnicas TCC) com fontes, e (c) sinaliza padroes de risco financeiro.

## Stack Implementada

| Componente | Ferramenta | Onde |
|------------|------------|------|
| LLM local | Ollama (llama3.2:1b) | localhost:11434 |
| Interface | Streamlit (chat + streaming) | localhost:8501 |
| Banco OLAP | DuckDB in-memory | in-process |
| Banco vetorial | ChromaDB persistido | `chroma_data/` |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) | in-process |
| ML basico | scikit-learn RandomForest | in-process |
| Linguagem | Python 3.13 | `.venv/` |

## Modulos Construidos

- [`app.py`](../app.py) - Streamlit. Chat com `st.write_stream`, sidebar com status do sistema e resumo financeiro, sessao em `st.session_state.messages`. Initialization cacheada via `@st.cache_resource`.
- [`config.py`](../config.py) - configuracoes Ollama, paths, parametros RAG (chunk_size=500, overlap=50, top_k=5), system prompt em portugues, mensagens UX.
- [`src/database.py`](../src/database.py) - DuckDB. Funcoes: `get_connection()`, `load_tables()`, `execute_query()` (bloqueia DROP/DELETE/UPDATE/INSERT), `get_schema_description()`, `get_summary()`.
- [`src/llm.py`](../src/llm.py) - cliente HTTP Ollama. Funcoes: `check_ollama()`, `chat_stream()`, `chat()`, `text_to_sql()` (com 5 exemplos few-shot e regras estritas), `rag_answer()` (responde apenas com base nos chunks fornecidos), `generate_collection_message()` (mensagem de cobranca empatica).
- [`src/rag.py`](../src/rag.py) - ChromaDB. Funcoes: `_read_file()` (txt/pdf), `_chunk_text()` (com smart breaks), `index_documents()`, `search()`, `is_indexed()`, `get_stats()`. Embeddings via SentenceTransformer.
- [`src/ml_model.py`](../src/ml_model.py) - RandomForest sklearn. Features: total_sessoes, taxa_atraso, taxa_nf, valor_sessao, modelo_encoded. Label: `taxa_atraso > 20%`. Funcoes: `train_model()`, `predict_risk()`, `get_risk_summary()`. **Limitacao reconhecida**: o label e funcao da feature taxa_atraso, gerando vazamento; corrigido na etapa 2 com modelo treinado em dataset publico.
- [`src/agents.py`](../src/agents.py) - roteamento por keywords. Funcao `route_question()` retorna uma de cinco categorias (financeiro/rag/risco/cobranca/geral). Cada agente e uma funcao Python que orquestra LLM + DB/RAG/ML.
- [`scripts/generate_data.py`](../scripts/generate_data.py) - gerador de dados sinteticos (seed=42).

## Dados Sinteticos

- 30 pacientes com codigos gregos (PAC-ALPHA a PAC-IRIS) - codificacao por etica.
- ~414 registros financeiros entre julho e dezembro de 2024.
- ~88% pagos, ~12% pendentes.
- 6 pacientes classificados como risco alto pelo modelo basico.
- Total pago: R$ 89.230. Total pendente: R$ 13.250.

## Documentos Indexados (RAG)

Em `docs/`:
- `codigo_etica_cfp.txt` - codigo de etica do CFP.
- `dsm5_resumo.txt` - resumo dos principais transtornos do DSM-5.
- `tecnicas_tcc.txt` - tecnicas de TCC para diferentes condicoes.

## Decisoes Tomadas

- **LLM local (Ollama)** em vez de API paga - alinha com a realidade financeira do publico-alvo (psicologa autonoma) e elimina dependencia de chave.
- **DuckDB in-memory** em vez de SQLite - performance OLAP melhor para os tipos de query agregada que dominam o caso (`SUM(valor)`, `GROUP BY mes`, etc.).
- **ChromaDB persistido em disco** - permite re-uso entre sessoes sem re-indexar a cada start.
- **Codificacao dos pacientes** (PAC-ALPHA, PAC-BETA, ...) - reforca compliance com sigilo profissional ainda em ambiente de demo.
- **Roteamento por keywords** em vez de LLM-as-router - barato, deterministico, suficiente para cinco categorias.
- **Streaming token-a-token** via `st.write_stream` - UX percebida muito melhor que aguardar resposta inteira.

## Arquivos NAO alteraveis pela etapa 2

Os seguintes itens foram entregues na etapa 1 e devem ser preservados:

- `data/pacientes.csv`, `data/financeiro.csv` (gerados com seed fixa).
- `scripts/generate_data.py` (mantido para reprodutibilidade).
- `src/database.py` (interface estavel; tools da Crew na etapa 2 reutilizam).
- `src/rag.py` logica de chunking/indexacao (apenas decoradores de observabilidade serao adicionados).
- `config.py` system prompt e mensagens UX.
- Comportamento do roteador por keywords (etapa 2 acrescenta camada nova mas nao remove o caminho existente).
