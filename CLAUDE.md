# AI Factory - Building Intelligent Systems (PUCPR)

## Sobre a matéria

Disciplina de faculdade (2º ano, PUCPR) focada na construção de sistemas de IA do zero até a interface.

## Stack de tecnologias

| Ferramenta | Propósito | URL local |
|------------|-----------|-----------|
| **Flowise** | Builder visual de fluxos de IA (no-code/low-code) | http://localhost:3000 |
| **Ollama** | LLM local (sem API key necessária) | http://localhost:11434 |
| **Streamlit** | Interfaces web em Python (dashboards, apps) | http://localhost:8501 |
| **Gradio** | Interfaces rápidas para demos de ML/chatbots | http://localhost:7860 |
| **Python 3.13** | Linguagem principal | — |

## Estrutura do projeto

```
AI Factory Building Intelligent Systems/
├── aula1/          ← Aula 1: Arquitetura + fluxos no Flowise
├── aula2/          ← Aula 2: Interfaces com Streamlit e Gradio
├── .env            ← Variáveis de ambiente (não commitar)
├── requirements.txt
└── CLAUDE.md
```

## Convenções

- **venv**: Ativar sempre antes de rodar Python: `.venv/Scripts/activate`
- **Flowise**: Iniciar com `npx flowise start` antes de acessar o browser
- **Ollama**: Deve estar rodando antes de conectar o Flowise
- **Variáveis de ambiente**: Usar `python-dotenv` e carregar do `.env`

## Aulas

- **Aula 1**: Design de arquitetura de sistemas de IA + primeiro fluxo no Flowise
- **Aula 2**: Interfaces web — UX writing, latência percebida, Streamlit vs Gradio

## Contexto do professor

- O professor usa **LM Studio** como alternativa ao Ollama
- Preferência do aluno: **Ollama** (já tem experiência)
- Sem API keys de cloud por enquanto — tudo local

## Notas de desenvolvimento

- Sempre preferir código limpo e simples (sem over-engineering)
- Aplicar conceitos de **UX writing**: mensagens amigáveis, feedback visual, streaming de respostas
- Implementar **streaming** nas respostas da IA para reduzir latência percebida
