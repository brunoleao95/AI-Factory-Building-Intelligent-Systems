# AI Factory - Building Intelligent Systems (PUCPR)

## Sobre a materia

Disciplina de faculdade (2o ano, PUCPR) focada na construcao de sistemas de IA do zero ate a interface.

## Stack de tecnologias

| Ferramenta | Proposito | URL local |
|------------|-----------|-----------|
| **Flowise** | Builder visual de fluxos de IA (no-code/low-code) | http://localhost:3000 |
| **Ollama** | LLM local (sem API key necessaria) | http://localhost:11434 |
| **Streamlit** | Interfaces web em Python (dashboards, apps) | http://localhost:8501 |
| **Gradio** | Interfaces rapidas para demos de ML/chatbots | http://localhost:7860 |
| **DuckDB** | Banco OLAP in-process (mini data warehouse) | -- |
| **ChromaDB** | Banco de dados vetorial (embeddings) | -- |
| **sentence-transformers** | Modelos de embedding locais (all-MiniLM) | -- |
| **Python 3.13** | Linguagem principal | -- |

## Estrutura do projeto

```
AI Factory Building Intelligent Systems/
├── .env            <- Variaveis de ambiente (nao commitar)
├── .gitignore
├── requirements.txt
└── CLAUDE.md
```

> Estrutura sera expandida quando o arquivo do projeto for enviado.

## Convencoes

- **venv**: Ativar sempre antes de rodar Python: `.venv/Scripts/activate`
- **Flowise**: Iniciar com `npx flowise start` antes de acessar o browser
- **Ollama**: Deve estar rodando antes de conectar o Flowise
- **Variaveis de ambiente**: Usar `python-dotenv` e carregar do `.env`
- **Git**: Repositorio inicializado, commitar progresso por aula

## Aulas

- **Aula 1**: Design de arquitetura de sistemas de IA + primeiro fluxo no Flowise
- **Aula 2**: Interfaces web -- UX writing, latencia percebida, Streamlit vs Gradio
- **Aula 3**: Memoria e contexto -- OLTP vs OLAP, DuckDB, Pandas, mini Data Warehouse, ETL, Text-to-SQL
- **Aula 4**: Embeddings, busca semantica, similaridade de cosseno, ChromaDB, preparacao para RAG
- **Aula 5** (proxima): RAG (Retrieval-Augmented Generation) -- combinar busca semantica com LLM

## Conceitos-chave por aula

### Aula 3 - Dados estruturados
- **OLTP** (PostgreSQL, MySQL): transacoes rapidas do dia a dia
- **OLAP** (DuckDB, BigQuery): analises sobre grandes volumes
- **DuckDB**: banco OLAP in-process, le CSV/Parquet/JSON direto, integra com Pandas
- **Text-to-SQL**: usuario pergunta em linguagem natural -> LLM traduz para SQL -> executa no DuckDB -> resposta formatada
- Padrao recomendado: DuckDB para carregar e agregar, Pandas para refinar e visualizar

### Aula 4 - Dados nao-estruturados
- **Embeddings**: representacao vetorial de texto (centenas/milhares de dimensoes)
- **Similaridade de cosseno**: mede distancia semantica entre vetores (1.0 = identico, 0.0 = nao relacionado)
- **ChromaDB**: banco vetorial para armazenar e buscar embeddings
- **Pipeline de indexacao**: carregar docs -> dividir em chunks -> gerar embeddings -> armazenar no ChromaDB
- **Pipeline de busca**: pergunta -> embedding -> busca por similaridade -> retorna top K documentos
- Proximo passo: combinar com LLM para criar RAG (aula 5)

## Contexto do professor

- O professor usa **LM Studio** como alternativa ao Ollama
- Preferencia do aluno: **Ollama** (ja tem experiencia)
- Sem API keys de cloud por enquanto -- tudo local

## Notas de desenvolvimento

- Sempre preferir codigo limpo e simples (sem over-engineering)
- Aplicar conceitos de **UX writing**: mensagens amigaveis, feedback visual, streaming de respostas
- Implementar **streaming** nas respostas da IA para reduzir latencia percebida
- Decisoes de arquitetura (OLTP vs OLAP, qual banco usar) sao importantes -- vibecoding nao resolve isso
