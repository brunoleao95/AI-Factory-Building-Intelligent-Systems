# Assistente de Gestao para Psicologa Clinica

> Projeto da disciplina **AI Factory - Building Intelligent Systems** (PUCPR, 2o ano)

## Demonstracao

- **Video demo (Etapa 2)**: https://www.youtube.com/watch?v=IyGmePDv7FU

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

- **LLM principal**: Ollama (`llama3.2:1b`) - local, sem API key. Usado em chat, Text-to-SQL, RAG e juiz DeepEval.
- **LLM dos agentes**: Ollama Cloud (`gpt-oss:20b-cloud`, free tier) - usado apenas pelo CrewAI porque function calling exige modelo 7B+.
- **Interface**: Streamlit
- **Banco estruturado**: DuckDB (OLAP in-process)
- **Banco vetorial**: ChromaDB
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **AutoML**: FLAML (XGBoost vencedor)
- **Agentes**: CrewAI
- **Observabilidade**: Langfuse Cloud
- **Avaliacao**: DeepEval (juiz Ollama local)
- **Linguagem**: Python 3.13

## Como rodar

### Pre-requisitos

- Python 3.10+
- Ollama instalado e rodando (`ollama serve`)
- Modelos baixados:
  - `ollama pull llama3.2:1b` (local, ~1.3 GB)
  - `ollama signin` + `ollama pull gpt-oss:20b-cloud` (Ollama Cloud free tier, necessario para a Crew)

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

# Gerar dados ficticios (uma vez)
python scripts/generate_data.py

# Treinar modelo de risco com FLAML (~3 min, baixa dataset UCI)
python scripts/train_ml_model.py
```

### Executar

```bash
# Iniciar Ollama (em outro terminal)
ollama serve

# Rodar a aplicacao
streamlit run app.py
```

Acesse http://localhost:8501

### Avaliacao (etapa 2)

```bash
# Executa o golden dataset (15 perguntas) e gera evals/results.md
python scripts/eval_deepeval.py
```

### Observabilidade (etapa 2)

1. Crie uma conta gratuita em https://cloud.langfuse.com
2. Crie um projeto e copie `Public Key` e `Secret Key`
3. Preencha no `.env`:
   - `LANGFUSE_PUBLIC_KEY=...`
   - `LANGFUSE_SECRET_KEY=...`
   - `LANGFUSE_HOST=https://cloud.langfuse.com` (ou `https://us.cloud.langfuse.com` se sua conta for region US)
4. Use o app normalmente - traces aparecem no dashboard com latencia e tokens.

Se as chaves nao estiverem no `.env`, o decorator de observabilidade vira no-op e o app continua funcionando offline.

### Documentos para RAG (opcional)

Coloque arquivos `.txt` ou `.pdf` na pasta `docs/` para habilitar a busca em documentos tecnicos. O sistema indexa automaticamente ao iniciar.

## Estrutura do Projeto

```
├── app.py                          # Interface Streamlit (ponto de entrada)
├── config.py                       # Configuracoes, constantes, system prompt
├── data/
│   ├── pacientes.csv               # Dados ficticios de pacientes (etapa 1)
│   ├── financeiro.csv              # Registros financeiros ficticios (etapa 1)
│   └── golden_dataset.json         # 15 perguntas para avaliacao (etapa 2)
├── docs/                           # Documentos para RAG (DSM-5, etica CFP, TCC)
├── src/
│   ├── database.py                 # DuckDB
│   ├── rag.py                      # ChromaDB
│   ├── llm.py                      # Ollama: chat, Text-to-SQL, RAG
│   ├── ml_model.py                 # Carrega FLAML.pkl, predict_risk
│   ├── agents.py                   # Roteador + agentes especializados
│   ├── crew.py                     # CrewAI: 3 agentes + tools (etapa 2)
│   └── observability.py            # Wrapper Langfuse (etapa 2)
├── scripts/
│   ├── generate_data.py            # Gera CSVs ficticios
│   ├── train_ml_model.py           # Treina FLAML sobre UCI (etapa 2)
│   └── eval_deepeval.py            # Roda golden dataset (etapa 2)
├── .env.example                    # Template de variaveis de ambiente
├── requirements.txt
├── CLAUDE.md                       # Referencia tecnica detalhada
└── README.md
```

## Exemplos de uso

- "Qual o faturamento total do mes de outubro?"
- "Quais pacientes estao inadimplentes?"
- "Qual o risco de inadimplencia do PAC-DELTA?"
- "Gerar mensagem de cobranca para PAC-PHI"
- "O que diz o DSM-5 sobre transtorno de ansiedade?" (requer docs indexados)
- "/equipe quem esta inadimplente e gere as mensagens de cobranca" (etapa 2 - CrewAI)
