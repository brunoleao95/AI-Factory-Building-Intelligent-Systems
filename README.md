# Assistente de Gestao para Psicologa Clinica

> Projeto da disciplina **AI Factory - Building Intelligent Systems** (PUCPR, 2o ano)

## Challenge-Based Learning (CBL)

- **Grande Ideia**: Gestao inteligente e produtividade tecnologica na psicologia clinica.
- **Pergunta Essencial**: Como a Inteligencia Artificial pode reduzir a carga administrativa e de pesquisa de psicologas autonomas, permitindo que elas foquem integralmente no acolhimento clinico?
- **Desafio**: Construir um Assistente Inteligente para Psicologas que centralize a gestao financeira (prevendo riscos de inadimplencia), automatize a geracao de mensagens de cobranca respeitosas e forneca um repositorio tecnico consultavel via RAG para suporte em sessoes e estudos.
- **Justificativa Pessoal**: *(a ser preenchida pelo aluno)*

## Funcionalidades

| Funcionalidade | Tecnologia | Descricao |
|----------------|------------|-----------|
| Consultas financeiras | DuckDB + Text-to-SQL | Perguntas em linguagem natural sobre pagamentos, faturamento, inadimplencia |
| Repositorio tecnico | ChromaDB + RAG | Busca semantica em documentos (DSM-5, TCC, Etica) |
| Analise de risco | scikit-learn | Classificacao de risco de inadimplencia por paciente |
| Cobranca WhatsApp | Ollama | Geracao de mensagens de cobranca respeitosas |
| Chat com streaming | Streamlit + Ollama | Interface conversacional com respostas em tempo real |

## Stack

- **LLM**: Ollama (llama3.2) - local, sem API key
- **Interface**: Streamlit
- **Banco estruturado**: DuckDB (OLAP in-process)
- **Banco vetorial**: ChromaDB
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **ML**: scikit-learn (RandomForestClassifier)
- **Linguagem**: Python 3.13

## Como rodar

### Pre-requisitos
- Python 3.10+
- Ollama instalado e rodando (`ollama serve`)
- Modelo llama3.2 baixado (`ollama pull llama3.2`)

### Instalacao

```bash
# Criar e ativar venv
python -m venv .venv
.venv/Scripts/activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Instalar dependencias
pip install -r requirements.txt

# Copiar variaveis de ambiente
cp .env.example .env

# Gerar dados ficticios
python scripts/generate_data.py
```

### Executar

```bash
# Iniciar Ollama (em outro terminal)
ollama serve

# Rodar a aplicacao
streamlit run app.py
```

Acesse: http://localhost:8501

### Documentos para RAG (opcional)

Coloque arquivos `.txt` ou `.pdf` na pasta `docs/` para habilitar a busca em documentos tecnicos. O sistema indexa automaticamente ao iniciar.

## Estrutura do Projeto

```
├── app.py                  # Interface Streamlit (ponto de entrada)
├── config.py               # Configuracoes e system prompt
├── data/
│   ├── pacientes.csv       # Dados ficticios de pacientes
│   └── financeiro.csv      # Registros financeiros ficticios
├── docs/                   # Documentos para RAG
├── src/
│   ├── __init__.py
│   ├── database.py         # DuckDB: carregar CSVs, queries SQL
│   ├── rag.py              # ChromaDB: indexacao e busca semantica
│   ├── llm.py              # Ollama: chat, Text-to-SQL, RAG
│   ├── ml_model.py         # scikit-learn: risco de inadimplencia
│   └── agents.py           # Roteamento e agentes especializados
├── scripts/
│   └── generate_data.py    # Gerador de dados ficticios
├── .env.example            # Template de variaveis de ambiente
├── requirements.txt
└── CLAUDE.md
```

## Exemplos de uso

- "Qual o faturamento total do mes de outubro?"
- "Quais pacientes estao inadimplentes?"
- "Qual o risco de inadimplencia do PAC-DELTA?"
- "Gerar mensagem de cobranca para PAC-PHI"
- "O que diz o DSM-5 sobre transtorno de ansiedade?" (requer docs indexados)
