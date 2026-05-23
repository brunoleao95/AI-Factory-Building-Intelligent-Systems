# AI Factory - Assistente de Gestao para Psicologa Clinica

## Sobre o Projeto

Disciplina **AI Factory - Building Intelligent Systems** (PUCPR, 2o ano).
Projeto final: assistente inteligente para psicologas autonomas que centraliza gestao financeira, fornece repositorio tecnico via RAG, prediz risco de inadimplencia com ML e orquestra tarefas compostas via equipe de agentes (CrewAI).

**CBL:**
- Grande ideia: Gestao inteligente e produtividade tecnologica na psicologia clinica
- Desafio: Assistente que centraliza gestao financeira + repositorio tecnico via RAG + camadas de inteligencia (ML, agentes, observabilidade, testes)

## Estado das Entregas

| Etapa | Status | Documento | O que adiciona |
|-------|--------|-----------|----------------|
| 1 | Entregue | [`entregas/etapa1.md`](entregas/etapa1.md) | Chat, RAG, Text-to-SQL, dados sinteticos, ML basico |
| 2 | Em andamento | [`entregas/etapa2.md`](entregas/etapa2.md) | FLAML AutoML (UCI), CrewAI (3 agentes), Langfuse, DeepEval |
| 3 | Pendente | _futuro_ | Producao: deploy, autenticacao, documentacao final |

## Stack

| Ferramenta | Proposito | Onde |
|------------|-----------|------|
| **Ollama** (llama3.2:1b local + gpt-oss:20b-cloud) | LLM duplo: 1b local para chat/RAG/SQL/juiz; 20b cloud para CrewAI (function calling) | localhost:11434 |
| **Streamlit** | Interface web do assistente | localhost:8501 |
| **DuckDB** | Banco OLAP in-memory (dados financeiros) | In-process |
| **ChromaDB** | Banco vetorial (embeddings para RAG) | Persistido em chroma_data/ |
| **sentence-transformers** | Embeddings locais (all-MiniLM-L6-v2) | In-process |
| **FLAML** | AutoML para risco de inadimplencia (etapa 2) | models/risco_inadimplencia.pkl |
| **CrewAI** | Multi-agente para tarefas compostas (etapa 2) | src/crew.py |
| **Langfuse** (cloud) | Observabilidade: traces, latencia, tokens (etapa 2) | cloud.langfuse.com |
| **DeepEval** | Suite de avaliacao com golden dataset (etapa 2) | scripts/eval_deepeval.py |
| **Python 3.13** | Linguagem principal | .venv/ |

## Estrutura do Projeto

```
AI Factory Building Intelligent Systems/
├── app.py                          <- Streamlit: interface principal (chat + streaming)
├── config.py                       <- Configuracoes, constantes, system prompt
├── data/
│   ├── pacientes.csv               <- 30 pacientes ficticios (etapa 1)
│   ├── financeiro.csv              <- ~414 registros (etapa 1)
│   ├── golden_dataset.json         <- 15 perguntas de avaliacao (etapa 2)
│   └── uci_credit_default.csv      <- Cache UCI (gitignored, baixado pelo train_ml_model)
├── docs/                           <- Corpus do RAG (NAO confundir com entregas/)
├── entregas/                       <- Relatorios tecnicos por etapa (separado de docs/)
│   ├── README.md
│   ├── etapa1.md                   <- Retrospectiva do projeto 1
│   └── etapa2.md                   <- Relatorio tecnico da etapa 2
├── evals/                          <- Resultados DeepEval (gerados, gitignored)
│   ├── results.json
│   └── results.md
├── models/                         <- Artefatos do FLAML (gerados, gitignored)
│   ├── risco_inadimplencia.pkl
│   └── metrics.json
├── src/
│   ├── __init__.py
│   ├── database.py                 <- DuckDB
│   ├── rag.py                      <- ChromaDB + Langfuse @observe
│   ├── llm.py                      <- Ollama + Langfuse @observe
│   ├── ml_model.py                 <- Carrega FLAML.pkl, predict_risk
│   ├── agents.py                   <- Roteador por keywords (etapa 1) + dispatch p/ Crew
│   ├── crew.py                     <- CrewAI: 3 agentes + tools (etapa 2)
│   └── observability.py            <- Wrapper Langfuse com fallback no-op (etapa 2)
├── scripts/
│   ├── generate_data.py            <- Gera CSVs ficticios (etapa 1)
│   ├── train_ml_model.py           <- Treina FLAML sobre UCI (etapa 2)
│   └── eval_deepeval.py            <- Roda golden dataset (etapa 2)
├── chroma_data/                    <- ChromaDB persistido (gitignored)
├── .env                            <- Secrets (gitignored)
├── .env.example                    <- Template do .env
├── .gitignore
├── requirements.txt
├── README.md
└── CLAUDE.md
```

## Arquitetura

### Fluxo Principal
1. Usuario digita pergunta no chat (Streamlit).
2. Se a pergunta tem prefixo `/equipe` ou e composta -> **Crew CrewAI** (3 agentes especializados).
3. Caso contrario, `agents.route_question()` detecta intencao por keywords e despacha para:
   - **financeiro** -> Text-to-SQL -> DuckDB -> LLM formata
   - **rag** -> ChromaDB busca semantica -> LLM responde com contexto
   - **risco** -> FLAML predict_risk -> formata analise
   - **cobranca** -> DuckDB busca pendencias -> LLM gera mensagem WhatsApp
   - **geral** -> LLM responde direto
4. Cada chamada LLM/RAG/Tool e instrumentada com `@observe` (Langfuse).
5. Resposta aparece com streaming (excecao: Crew retorna tudo de uma vez).

### Dados Estruturados (DuckDB)
- **pacientes**: id_paciente, nome_codigo (PAC-ALPHA etc), valor_sessao, modelo_cobranca
- **financeiro**: id_registro, id_paciente, data_sessao, valor, status_pagamento, nf_emitida
- Relacionamento: `financeiro.id_paciente -> pacientes.id_paciente`
- Seguranca: queries destrutivas bloqueadas (DROP, DELETE, UPDATE, INSERT)

### Pipeline RAG (ChromaDB)
- Indexacao: ler docs/ -> chunking (500 chars, 50 overlap) -> embeddings -> ChromaDB
- Busca: query -> embedding -> cosine similarity -> top 5 chunks
- Suporta .txt e .pdf (via pypdf)
- `@observe(name="rag_retrieval")` registra cada busca como span no Langfuse

### Modelo ML (FLAML AutoML, etapa 2)
- Dataset: **UCI Default of Credit Card Clients** (id 350, 30000 registros).
- Treinado offline pelo `scripts/train_ml_model.py` (XGBoost via FLAML, 60s budget).
- Features mapeadas para o contexto psicologia: LIMIT_BAL = total_sessoes * valor_sessao, PAY_X = atrasos por mes, BILL/PAY_AMT_X = valor faturado/pago por mes.
- Metricas no test set: ver `models/metrics.json`.
- Classifica pacientes em ALTO (>= 50%), MEDIO (>= 30%), BAIXO.

### Agentes CrewAI (etapa 2)
Tres agentes orquestrados em sequence quando a pergunta e composta:
- **Analista Financeiro**: tools `query_finance`, `financial_summary`.
- **Especialista Clinico**: tool `search_documents` (RAG).
- **Analista de Risco**: tools `predict_patient_risk`, `draft_collection_message`, `query_finance`.

Acionados por: prefixo `/equipe`/`/crew` OU heuristica de pergunta composta (multiplas intencoes em uma frase).

**Modelo dedicado**: a Crew usa `OLLAMA_CREW_MODEL` (default `gpt-oss:20b-cloud` via Ollama Cloud free tier), enquanto o resto do app usa `OLLAMA_MODEL` (`llama3.2:1b` local). A separacao foi necessaria porque function calling exige modelo 7B+; a 1.2B emite JSON malformado que vira ruido. O modelo cloud responde em ~30s para a Crew completa contra ~218s da 1B, com qualidade de output muito superior.

### Observabilidade (Langfuse, etapa 2)
- `src/observability.py` instancia cliente Langfuse e expoe decorator `@observe`.
- Se `.env` nao tiver `LANGFUSE_PUBLIC_KEY` e `LANGFUSE_SECRET_KEY` o decorator vira no-op.
- Spans registrados: `process_question` (top), `rag_retrieval`, `text_to_sql`, `ollama_chat_stream`, `rag_answer`, `generate_collection_message`, `crew_run`.

### Avaliacao (DeepEval, etapa 2)
- 15 perguntas em `data/golden_dataset.json` (financeiro, RAG, risco, cobranca, fora-de-escopo).
- Suite roda `process_question` para cada pergunta e avalia com `FaithfulnessMetric` + `AnswerRelevancyMetric`.
- Juiz: Ollama llama3.2 (sem API key paga).
- Resultados em `evals/results.md` (referenciados no relatorio etapa 2).

### Interface (Streamlit)
- Chat com `st.chat_message` + `st.write_stream` (streaming real)
- Session state para historico
- Sidebar mostra: Ollama, dados, RAG, modelo ML (com ROC-AUC), Langfuse, resumo financeiro
- Feedback visual por tipo de agente (incluindo "Equipe de agentes coordenando...")

## Convencoes

- **venv**: `.venv/Scripts/activate` (Windows)
- **Ollama**: Deve estar rodando antes de iniciar o app
- **Variaveis**: python-dotenv carrega do .env
- Codigo limpo, sem over-engineering
- Nomes de pacientes sempre codificados (etica)
- Valores formatados em R$

## Como Rodar

```bash
# 1. Ativar venv e instalar deps
.venv/Scripts/activate
pip install -r requirements.txt

# 2. Configurar .env (copiar de .env.example e preencher Langfuse se quiser observabilidade)
cp .env.example .env

# 3. Iniciar Ollama (outro terminal)
ollama serve

# 4. Gerar dados sinteticos (uma vez)
python scripts/generate_data.py

# 5. Treinar modelo ML (uma vez, ~3 min)
python scripts/train_ml_model.py

# 6. Rodar app
streamlit run app.py

# 7. (Opcional) Rodar avaliacao
python scripts/eval_deepeval.py
```

## O que NAO alterar (preservar entre etapas)

- `data/pacientes.csv`, `data/financeiro.csv` (gerados na etapa 1, seed fixa).
- `scripts/generate_data.py`.
- Schema do DuckDB (`src/database.py:get_schema_description`) e funcoes publicas (`load_tables`, `execute_query`, `get_summary`).
- Comportamento do roteador por keywords (`route_question`) - a etapa 2 acrescenta camada nova mas preserva o caminho existente.
- System prompt e mensagens UX em `config.py` (apenas adicoes).
- Lista de keywords nos arrays `KEYWORDS_*` em `src/agents.py`.

## Dados Gerados (etapa 1, mantidos)

- 30 pacientes com codigos gregos (PAC-ALPHA ate PAC-IRIS)
- 414 registros financeiros (jul-dez 2024)
- ~88% pagos, ~12% pendentes
- Total pago: R$ 89.230, Total pendente: R$ 13.250
