# AI Factory - Assistente de Gestao para Psicologa Clinica

## Sobre o Projeto

Disciplina **AI Factory - Building Intelligent Systems** (PUCPR, 2o ano).
Projeto final: assistente inteligente para psicologas autonomas que centraliza gestao financeira e fornece repositorio tecnico via RAG.

**CBL:**
- Grande ideia: Gestao inteligente e produtividade tecnologica na psicologia clinica
- Desafio: Assistente que centraliza gestao financeira + repositorio tecnico via RAG

## Stack

| Ferramenta | Proposito | Onde |
|------------|-----------|------|
| **Ollama** (llama3.2) | LLM local, chat, Text-to-SQL, RAG | localhost:11434 |
| **Streamlit** | Interface web do assistente | localhost:8501 |
| **DuckDB** | Banco OLAP in-memory (dados financeiros) | In-process |
| **ChromaDB** | Banco vetorial (embeddings para RAG) | Persistido em chroma_data/ |
| **sentence-transformers** | Embeddings locais (all-MiniLM-L6-v2) | In-process |
| **scikit-learn** | Modelo ML de risco de inadimplencia | In-process |
| **Python 3.13** | Linguagem principal | .venv/ |

## Estrutura do Projeto

```
AI Factory Building Intelligent Systems/
├── app.py                  <- Streamlit: interface principal (chat + streaming)
├── config.py               <- Configuracoes, constantes, system prompt, mensagens UX
├── data/
│   ├── pacientes.csv       <- 30 pacientes ficticios (id, codigo, valor, modelo cobranca)
│   └── financeiro.csv      <- ~414 registros (sessoes, pagamentos, NFs)
├── docs/                   <- Documentos para RAG (usuario fornece .txt/.pdf)
│   └── README.md
├── src/
│   ├── __init__.py
│   ├── database.py         <- DuckDB: load_tables(), execute_query(), get_schema_description()
│   ├── rag.py              <- ChromaDB: index_documents(), search(), chunking
│   ├── llm.py              <- Ollama: chat_stream(), text_to_sql(), rag_answer()
│   ├── ml_model.py         <- scikit-learn: train_model(), predict_risk(), RandomForest
│   └── agents.py           <- Roteador + agentes (financeiro, repositorio, risco, cobranca)
├── scripts/
│   └── generate_data.py    <- Gera CSVs ficticios (seed=42, reproduzivel)
├── chroma_data/            <- ChromaDB persistido (no .gitignore)
├── .env                    <- Variaveis de ambiente (nao commitar)
├── .env.example            <- Template do .env
├── .gitignore
├── requirements.txt
├── README.md
└── CLAUDE.md
```

## Arquitetura dos Componentes

### Fluxo Principal
1. Usuario digita pergunta no chat (Streamlit)
2. `agents.route_question()` detecta intencao por keywords
3. Despacha para agente adequado:
   - **financeiro** → Text-to-SQL → DuckDB → LLM formata resposta
   - **rag** → ChromaDB busca semantica → LLM responde com contexto
   - **risco** → scikit-learn predict_risk() → formata analise
   - **cobranca** → DuckDB busca pendencias → LLM gera mensagem WhatsApp
   - **geral** → LLM responde direto
4. Resposta aparece com streaming (token por token)

### Dados Estruturados (DuckDB)
- **pacientes**: id_paciente, nome_codigo (PAC-ALPHA etc), valor_sessao, modelo_cobranca
- **financeiro**: id_registro, id_paciente, data_sessao, valor, status_pagamento, nf_emitida
- Relacionamento: financeiro.id_paciente → pacientes.id_paciente
- Seguranca: queries destrutivas bloqueadas (DROP, DELETE, UPDATE, INSERT)

### Pipeline RAG (ChromaDB)
- Indexacao: ler docs/ → chunking (500 chars, 50 overlap) → embeddings → ChromaDB
- Busca: query → embedding → cosine similarity → top 5 chunks
- Suporta .txt e .pdf (via pypdf)

### Modelo ML (scikit-learn)
- RandomForestClassifier para risco de inadimplencia
- Features: taxa_atraso, taxa_nf, total_sessoes, valor_sessao, modelo_cobranca
- Label: risco alto se taxa_atraso > 20%
- Treinado automaticamente ao carregar dados

### Interface (Streamlit)
- Chat com st.chat_message + st.write_stream (streaming real)
- Session state para historico
- Sidebar com status do sistema + resumo financeiro
- Feedback visual por tipo de agente ("Consultando dados financeiros...")

## Convencoes

- **venv**: `.venv/Scripts/activate` (Windows)
- **Ollama**: Deve estar rodando antes de iniciar o app
- **Variaveis**: python-dotenv carrega do .env
- Codigo limpo, sem over-engineering
- Nomes de pacientes sempre codificados (etica)
- Valores formatados em R$

## Como Rodar

```bash
# 1. Ativar venv
.venv/Scripts/activate

# 2. Iniciar Ollama (outro terminal)
ollama serve

# 3. Gerar dados (se necessario)
python scripts/generate_data.py

# 4. Rodar app
streamlit run app.py
```

## Dados Gerados

- 30 pacientes com codigos gregos (PAC-ALPHA ate PAC-IRIS)
- 414 registros financeiros (jul-dez 2024)
- ~88% pagos, ~12% pendentes
- 6 pacientes classificados como risco alto
- Total pago: R$ 89.230, Total pendente: R$ 13.250
