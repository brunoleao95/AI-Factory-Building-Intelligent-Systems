# Assistente de Gestao para Psicologa Clinica

> Projeto da disciplina **AI Factory - Building Intelligent Systems** (PUCPR, 2o ano)

## Challenge-Based Learning (CBL)

- **Grande Ideia**: Gestao inteligente e produtividade tecnologica na psicologia clinica.
- **Pergunta Essencial**: Como a Inteligencia Artificial pode reduzir a carga administrativa e de pesquisa de psicologas autonomas, permitindo que elas foquem integralmente no acolhimento clinico?
- **Desafio**: Construir um Assistente Inteligente para Psicologas que centralize a gestao financeira (prevendo riscos de inadimplencia), automatize a geracao de mensagens de cobranca respeitosas e forneca um repositorio tecnico consultavel via RAG para suporte em sessoes e estudos.
- **Justificativa Pessoal**: Acompanho de perto o dia a dia de uma psicóloga e percebo quanto tempo ela gasta com controle financeiro, cobrança de pacientes e busca em materiais técnicos. Tempo que poderia ser dedicado ao atendimento clínico, estudo da psicologia ou até mesmo descanso. Esse projeto nasceu da vontade de resolver um problema real que vejo acontecer, unindo minha formação em IA com uma necessidade concreta de quem trabalha sozinha na área da saúde mental.

## Funcionalidades

| Funcionalidade | Tecnologia | Etapa | Descricao |
|----------------|------------|-------|-----------|
| Consultas financeiras | DuckDB + Text-to-SQL | 1 | Perguntas em linguagem natural sobre pagamentos, faturamento, inadimplencia |
| Repositorio tecnico | ChromaDB + RAG | 1 | Busca semantica em documentos (DSM-5, TCC, Etica) |
| Cobranca WhatsApp | Ollama | 1 | Geracao de mensagens de cobranca respeitosas |
| Chat com streaming | Streamlit + Ollama | 1 | Interface conversacional com respostas em tempo real |
| Analise de risco | FLAML + UCI dataset | 2 | Classificacao de risco de inadimplencia (XGBoost via AutoML) |
| Equipe de agentes | CrewAI | 2 | 3 agentes especializados para tarefas compostas (`/equipe`) |
| Observabilidade | Langfuse | 2 | Traces, latencia, tokens (cloud) |
| Avaliacao | DeepEval | 2 | 15 perguntas golden + faithfulness/answer_relevancy |

## Stack

- **LLM**: Ollama (llama3.2:1b) - local, sem API key
- **Interface**: Streamlit
- **Banco estruturado**: DuckDB (OLAP in-process)
- **Banco vetorial**: ChromaDB
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **AutoML**: FLAML (XGBoost vencedor)
- **Agentes**: CrewAI
- **Observabilidade**: Langfuse Cloud
- **Avaliacao**: DeepEval (juiz Ollama)
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

# Copiar variaveis de ambiente (preencha LANGFUSE_* se quiser observabilidade)
cp .env.example .env

# Gerar dados ficticios
python scripts/generate_data.py

# Treinar modelo de risco com FLAML (etapa 2) - ~3 minutos, baixa dataset UCI
python scripts/train_ml_model.py
```

### Executar

```bash
# Iniciar Ollama (em outro terminal)
ollama serve

# Rodar a aplicacao
streamlit run app.py
```

### Avaliacao (etapa 2)

```bash
# Executa o golden dataset (15 perguntas) e gera evals/results.md
python scripts/eval_deepeval.py
```

### Observabilidade (etapa 2)

1. Crie uma conta gratuita em https://cloud.langfuse.com
2. Crie um projeto e copie `Public Key` e `Secret Key`
3. Preencha `LANGFUSE_PUBLIC_KEY` e `LANGFUSE_SECRET_KEY` no `.env`
4. Use o app normalmente - traces aparecem no dashboard com latencia e tokens.

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
- "/equipe quem esta inadimplente e gere as mensagens de cobranca" (etapa 2 - CrewAI)

## Documentacao

- [`CLAUDE.md`](CLAUDE.md) - referencia tecnica completa para agentes de IA
- [`entregas/etapa1.md`](entregas/etapa1.md) - retrospectiva do projeto 1
- [`entregas/etapa2.md`](entregas/etapa2.md) - relatorio tecnico da etapa 2 (em construcao)
