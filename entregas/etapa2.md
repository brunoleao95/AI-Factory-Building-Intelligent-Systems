# Etapa 2 — Camadas de Inteligência

> Relatório técnico da Etapa 2 do CBL (fase **Investigar**). Documenta a inclusão das quatro camadas exigidas pela rubrica: modelo de ML treinado, agentes CrewAI, observabilidade Langfuse e suite de testes DeepEval. Baseado no estado do sistema descrito em [`etapa1.md`](etapa1.md), preservando todo o trabalho da etapa anterior.

## 1. Visão Geral

A Etapa 1 entregou um chatbot funcional para uma psicóloga clínica autônoma — chat com streaming, Text-to-SQL sobre DuckDB, RAG sobre ChromaDB, classificador básico de risco e gerador de mensagens de cobrança. A Etapa 2 acrescenta **inteligência preditiva** (AutoML), **autonomia operacional** (agentes), **transparência operacional** (observabilidade) e **garantia de qualidade** (testes), formando o ciclo completo de um sistema inteligente em produção.

Decisões deliberadas:
- **Aditividade**: nada do que existia foi removido. O roteador por keywords da Etapa 1 continua decidindo perguntas simples; perguntas compostas migram para a Crew.
- **Coerência de stack**: tudo continua local (Ollama, DuckDB, ChromaDB) salvo o Langfuse Cloud (free tier).
- **Custo zero de inferência**: o juiz do DeepEval é a própria llama3.2 — nenhuma API paga é necessária para rodar o ciclo completo.

## 2. Modelo de ML

### 2.1. Dataset

Foi escolhido o **UCI Default of Credit Card Clients** (Yeh & Lien, 2009), `id=350` no UCI ML Repository, com **30.000 registros** e 23 features. O download é automático via pacote `ucimlrepo` no script `scripts/train_ml_model.py`.

**Conexão com o domínio**: o problema estrutural é o mesmo do consultório — predizer, a partir de uma série de pagamentos esperados e do histórico recente, quem irá inadimplir. As variáveis foram mapeadas para o vocabulário psicológico:

| UCI | Significado original | Análogo no consultório |
|-----|----------------------|------------------------|
| `LIMIT_BAL` | limite de crédito | total esperado (`total_sessoes × valor_sessao`) |
| `PAY_0` … `PAY_6` | meses de atraso | atrasos em sessões nos últimos 6 meses |
| `BILL_AMT1` … `BILL_AMT3` | fatura mensal | valor faturado por mês |
| `PAY_AMT1` … `PAY_AMT3` | pagamento mensal | valor recebido por mês |
| `AGE` | idade | usado o valor mediano de adultos em terapia (35) |

Variável-alvo: `default_payment_next_month` (1 = inadimplência, 0 = adimplente).

### 2.2. Treinamento (FLAML AutoML)

O pipeline foi:
1. Train/test split estratificado 80/20, `random_state=42` → 24.000 treino, 6.000 teste.
2. `flaml.AutoML().fit(... task="classification", metric="roc_auc", time_budget=60s, estimator_list=["lgbm", "rf", "xgboost", "extra_tree"], eval_method="cv", n_splits=3)`.
3. Avaliação no conjunto de teste, persistência em `models/risco_inadimplencia.pkl` + `models/metrics.json`.

O FLAML escolheu **XGBoost** com 644 árvores e learning rate `0.0099`. O modelo é carregado em runtime por `src/ml_model.py` e suas predições alimentam o agente de risco do chat (que mantém a mesma assinatura `predict_risk(id_paciente)` da Etapa 1).

### 2.3. Métricas

| Métrica | Valor (test set) |
|---------|------------------|
| Accuracy | 0,8192 |
| Precision | 0,6667 |
| Recall | 0,3647 |
| F1 | 0,4715 |
| **ROC-AUC** | **0,7775** |

**Métrica prioritária no contexto do consultório**: **recall** (com ROC-AUC como medida secundária). A justificativa é direta:
- **Falso negativo** (modelo diz "vai pagar" mas o paciente não paga): a psicóloga é surpreendida por uma sessão sem pagamento — perda financeira efetiva, sem chance de mitigar.
- **Falso positivo** (modelo diz "vai atrasar" mas o paciente pagaria normalmente): a psicóloga recebe um alerta que não vira ação destrutiva — no máximo, uma mensagem extra de cobrança que o sistema gera com tom empático.

Como o custo do FN é assimetricamente maior, o recall importa mais que a precisão. O recall atual (0,36) é um ponto fraco assumido — a Etapa 3 pode endereçar isso ajustando o limiar de decisão (`probability >= 0.30` em vez de `0.50`) ou re-balanceando classes. O ROC-AUC de 0,78 mostra que o modelo separa bem as classes; só está sendo conservador na fronteira.

## 3. Agentes CrewAI

A Etapa 1 já tinha "agentes" em formato de funções Python despachadas por keywords. A Etapa 2 introduz **agentes autônomos com tools, papéis e backstories**, na linha do que o material da Aula 6 propõe.

### 3.1. Agentes implementados

| Agente | Papel | Tools |
|--------|-------|-------|
| **Analista Financeiro Clínico** | Investigar pagamentos, pendências, faturamento e nota fiscal usando dados reais. | `query_finance` (Text-to-SQL → DuckDB), `financial_summary` (resumo consolidado) |
| **Especialista Clínico** | Esclarecer dúvidas técnicas (DSM-5, código de ética, TCC) com fonte citada. | `search_documents` (RAG ChromaDB) |
| **Analista de Risco e Cobrança** | Prever inadimplência via FLAML e gerar mensagens de cobrança empáticas. | `predict_patient_risk`, `draft_collection_message`, `query_finance` |

Cada tool é uma fachada fina sobre funções já existentes — **não há duplicação de lógica entre o roteador da Etapa 1 e os agentes**. Cada agente tem `max_iter=4-5` (boa prática mencionada na Aula 6 sobre custo de chamadas em loop).

### 3.2. Orquestração

Crew em `Process.sequential` com três tarefas:
1. **Plano** (Analista Financeiro): identifica subtarefas e atribui agentes.
2. **Investigar dados** (Analista de Risco): coleta dados/risco/pendências.
3. **Consolidar** (Especialista Clínico): redige resposta final em português, incluindo mensagens de cobrança quando solicitado.

A LLM dos agentes é o mesmo Ollama llama3.2 (via `crewai.LLM(model="ollama/llama3.2:1b", ...)` — LiteLLM por baixo).

### 3.3. Trigger no chat

A Crew **não** substitui o roteador. O dispatcher em `src/agents.py:process_question` aciona a Crew somente quando:
- a pergunta começa com `/equipe` ou `/crew`, **ou**
- a heurística `should_use_crew` detecta uma pergunta composta (presença de conectivos como "e", "depois", "também" combinados com verbos de ação como "gere", "envie", "cobra").

Caso contrário, o roteador rápido por keywords da Etapa 1 continua decidindo. Esse desenho minimiza latência e custo: o usuário só "paga" o overhead da Crew quando o ganho de orquestração compensa.

### 3.4. Decisão de modelo (achado iterativo)

A primeira execução da Crew com `llama3.2:1b` (mesmo modelo do resto do app) produziu output completamente inutilizável: o modelo, em vez de invocar as tools via function calling, **gerava texto descrevendo o formato de tool call** (`{function Busca semantica {object} ...`). É um failure mode conhecido: function calling exige modelos a partir de ~7B parâmetros, e a 1.2B só era treinada para chat livre.

A decisão foi separar os modelos:

| Variável | Modelo | Onde é usado |
|----------|--------|--------------|
| `OLLAMA_MODEL` | `llama3.2:1b` (local) | Chat normal, Text-to-SQL, RAG answer, juiz DeepEval |
| `OLLAMA_CREW_MODEL` | `gpt-oss:20b-cloud` (Ollama Cloud, free tier) | Apenas os agentes CrewAI |

O `gpt-oss:20b-cloud` foi escolhido por estar no tier gratuito do Ollama Cloud, ser grande o suficiente para function calling robusto, e seguir formato OpenAI (alinhado com LiteLLM). Resultado da mesma pergunta `/equipe quem está inadimplente e gere mensagem para o paciente de maior risco`:

| Métrica | `llama3.2:1b` | `gpt-oss:20b-cloud` |
|---------|---------------|---------------------|
| Latência | 218s | **~30s** (7× mais rápido) |
| Tools efetivamente chamadas | 0 (alucinou JSON) | 2-3 por execução |
| Códigos de paciente retornados | inventados | reais (PAC-MU, PAC-EPSILON, …) |
| Output utilizável para o usuário | Não | Sim |

Ainda foi necessário **endurecer os backstories** dos agentes (deixando explícito "NUNCA invente nomes pessoais; SEMPRE chame a tool query_finance") e **simplificar a sequência de tasks** (3 tasks com contexto encadeado em vez de uma task de planejamento + execução + consolidação). A decisão de manter o `llama3.2:1b` no resto do app preserva a filosofia "100% local" para o caminho rápido — a Crew é a única que paga o trade-off de cloud, e mesmo assim no tier gratuito.

## 4. Observabilidade — Langfuse

### 4.1. Setup

- Hospedagem: **Langfuse Cloud** (free tier, `cloud.langfuse.com`).
- Configuração: três variáveis no `.env` (`LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`).
- `src/observability.py` instancia o cliente, expõe um decorator `@observe` próprio do projeto e — quando as chaves estão ausentes — degrada graciosamente para um decorator no-op (o app local continua funcionando sem credenciais).

### 4.2. Spans instrumentados

| Span | Onde | O que captura |
|------|------|---------------|
| `process_question` | [`src/agents.py:255`](../src/agents.py:255) | Pergunta original do usuário, agente escolhido — trace raiz |
| `rag_retrieval` | [`src/rag.py:174`](../src/rag.py:174) | Query, top-k, chunks recuperados |
| `text_to_sql` | [`src/llm.py:74`](../src/llm.py:74) | Pergunta NL → SQL gerado |
| `ollama_chat_stream` | [`src/llm.py:21`](../src/llm.py:21) | `as_type="generation"`, modelo, n_messages — registra latência e tokens |
| `ollama_chat` | [`src/llm.py:66`](../src/llm.py:66) | Versão não-streaming usada por `text_to_sql` e `generate_collection_message` |
| `rag_answer` | [`src/llm.py:126`](../src/llm.py:126) | Pergunta + chunks → resposta |
| `generate_collection_message` | [`src/llm.py:158`](../src/llm.py:158) | Template empático WhatsApp |
| `crew_run` | [`src/crew.py:233`](../src/crew.py:233) | Execução completa da Crew |

Após cada turno do chat, `observability.flush()` é chamado para empurrar os eventos para o servidor.

### 4.3. Capturas de tela

Foram realizadas 6 chamadas reais ao pipeline (3 financeiras, 1 RAG, 1 risco, 1 cobrança), gerando os traces salvos em `cloud.langfuse.com`:

- [`Screenshot_1.png`](Screenshot_1.png) — **lista de traces**: 6 chamadas a `process_question` com sub-spans aninhados (`text_to_sql`, `rag_retrieval`, `rag_answer`, `ollama_chat`, `ollama_chat_stream`). Atende ao critério de "5+ traces" da rubrica.
- [`Screenshot_2.png`](Screenshot_2.png) — **detalhe de um `process_question`**: mostra input/output/metadata. A latência aparece 0.00s porque a função decorada retorna um generator imediatamente — o tempo real de execução é capturado nos sub-spans aninhados.
- [`Screenshot_3.png`](Screenshot_3.png) — **detalhe de um `ollama_chat_stream`**: latência **3.25s** explícita, input/output completos do LLM. É a captura que materializa o requisito da rubrica de "latência, tokens e custo".

> **Como reproduzir**: rode `streamlit run app.py`, faça 5+ perguntas variadas, abra `cloud.langfuse.com` → Tracing. Os traces aparecem em ordem cronológica reversa com latência, tokens e árvore de spans expansível.

## 5. Avaliação — Golden Dataset + DeepEval

### 5.1. Golden dataset

`data/golden_dataset.json` contém **15 perguntas** distribuídas por categoria:

| Categoria | Qtde | Exemplos |
|-----------|------|----------|
| Financeiro | 4 | "Quais pacientes estão pendentes?", "Quanto recebi em outubro?" |
| RAG | 4 | "Técnicas de TCC para ansiedade", "Por quanto tempo guardar prontuários?" |
| Risco | 3 | "Qual paciente tem maior risco?", "Risco de PAC-ALPHA?" |
| Cobrança | 2 | "Gere mensagem para PAC-ALPHA", composta para Crew |
| Fora-de-escopo | 2 | "Capital da França?", "Me ensine Python" |

Cada item tem `expected_substrings` (sanity check de presença), `expected_route` (qual agente deveria responder) e `context_hint` (auditoria humana).

### 5.2. Suite DeepEval

`scripts/eval_deepeval.py` faz:
1. Carrega DuckDB e indexa RAG.
2. Para cada pergunta, executa `process_question` (mesmo caminho do app real).
3. Captura resposta + chunks recuperados.
4. Avalia com `FaithfulnessMetric` (resposta fiel ao contexto?) e `AnswerRelevancyMetric` (resposta endereça a pergunta?).
5. Persiste resultados em `evals/results.json` + `evals/results.md`.

**Juiz**: a própria llama3.2 via subclasse customizada de `DeepEvalBaseLLM` — alinha a stack (sem API paga), aceita o pequeno custo de consistência inferior a um GPT-4.

### 5.3. Resultados

Execução completa do `scripts/eval_deepeval.py` em 09/05/2026. Resultados salvos em [`evals/results.json`](../evals/results.json) e [`evals/results.md`](../evals/results.md).

| Métrica | Valor |
|---------|-------|
| Perguntas avaliadas | **15** |
| Faithfulness média | **0,91** |
| Answer Relevancy média | **0,74** |
| Aprovadas (≥ 0,5) | **15/15** em ambas as métricas |
| Latência média de geração | **17,7s** (puxada por `cob-02` que aciona a Crew, 218s) |

| Categoria | Faithfulness | Relevancy | Observação |
|-----------|--------------|-----------|------------|
| Financeiro (4 itens) | 1,00 | 0,79 | Forte: SQL bem estruturado nos casos típicos |
| RAG (4 itens) | 1,00 | 0,67 | Forte em fidelidade aos chunks; relevância média devido a um bug de roteamento (ver 5.4) |
| Risco (3 itens) | 0,83 | 0,78 | Modelo FLAML produz tabela consistente |
| Cobrança (2 itens) | 0,75 | 0,50 | Caso composto com Crew levou 218s mas chegou ao output esperado |
| Fora-de-escopo (2 itens) | 0,83 | 1,00 | Sistema sinaliza limites bem, mas poderia ser mais explícito |

### 5.4. Análise dos pontos fracos identificados

A inspeção manual dos itens com score ≤ 0,67 revelou dois problemas distintos:

**(a) Bug real do roteador (alta prioridade)**

A pergunta `rag-02` ("Por quanto tempo um psicólogo deve guardar prontuários segundo o CFP?") deveria ter sido roteada para o agente RAG. Em vez disso, foi para o agente financeiro porque a função `route_question` ([`src/agents.py:36`](../src/agents.py:36)) deu peso à palavra "tempo" e ao verbo "guardar" como sinais financeiros. O resultado foi um SQL inválido:

```
Erro na consulta SQL: Binder Error: Referenced table "T2" not found!
```

Esse mesmo bug também explica o output alucinado visível no Screenshot_3 do Langfuse (resposta sobre "366 dias" para a mesma pergunta). O caso `cob-01` ("Gere uma mensagem de cobrança para PAC-ALPHA") também esbarrou em SQL inválido por outro caminho do mesmo roteador.

> **Melhoria proposta para a etapa 3**: substituir as keywords financeiras genéricas ("tempo", "guardar") por matchers mais específicos (ex.: prefixo `R$`, palavras como "outubro", "faturamento"), ou — solução mais robusta — substituir o roteador por um classificador LLM-as-router com cinco categorias e escolha por maior probabilidade.

**(b) Limitação do juiz llama3.2:1b**

Em vários itens com score 0,50 (especialmente `cob-01` faithfulness e `risco-02` faithfulness), o motivo registrado pelo juiz é incoerente — cita "Einstein won the Nobel Prize" (texto-template do DeepEval que vazou para a saída) ou produz frases vazias como justificativa. A llama3.2:1b é forte em conversação direta, mas perde foco quando precisa raciocinar sobre prompts longos com schema JSON estruturado.

> **Melhoria proposta para a etapa 3**: testar `ollama pull llama3.2` (modelo de 3B parâmetros — 4× maior, ainda local e gratuito) ou `llama3.1:8b` como juiz dedicado. Mantém-se a llama3.2:1b para o uso interativo do app, onde a velocidade conta mais que a precisão estrutural.

**(c) Outras melhorias incrementais**

- Aumentar `RAG_TOP_K` de 5 para 7 nas perguntas RAG mais difíceis (sigilo, transtornos específicos), facilitando ao LLM citar evidência mais variada.
- Adicionar mais exemplos few-shot ao prompt de `text_to_sql` ([`src/llm.py:74`](../src/llm.py:74)) para casos com agrupamento temporal (atualmente são 5; subir para 8-10).
- Reforçar o system prompt com cláusula explícita sobre escopo: "Se a pergunta não for sobre gestão clínica de psicologia, responda que está fora do seu escopo e ofereça redirecionamento."

## 6. O que fica preservado da Etapa 1

Conforme combinado, nada da Etapa 1 foi removido ou alterado em comportamento:
- `data/pacientes.csv`, `data/financeiro.csv` (gerados com seed fixa).
- `scripts/generate_data.py`.
- Schema do DuckDB e funções públicas em `src/database.py`.
- Roteador por keywords (`route_question`) — apenas ganhou um wrapper que detecta perguntas compostas antes do dispatch.
- System prompt e mensagens UX em `config.py` — apenas adições (mensagem nova `MSG_EQUIPE_TRABALHANDO`, exemplo `/equipe` na boas-vindas).

A retrospectiva detalhada da Etapa 1 está em [`etapa1.md`](etapa1.md).

## 7. Reflexão sobre o processo

- **O que deu certo**: a separação clara de camadas da Etapa 1 (LLM, DB, RAG, ML, agents) facilitou enormemente plugar tools de CrewAI sem reescrever lógica. O decorator no-op do Langfuse permitiu instrumentar tudo sem forçar credenciais durante o desenvolvimento. A separação `OLLAMA_MODEL` × `OLLAMA_CREW_MODEL` deu para reusar todo o resto do app sem trocar uma linha de código.
- **O que deu errado / surpresas**: (a) `ucimlrepo` retorna `X1..X23` em vez de nomes canônicos — ciclo de re-treinamento gasto. (b) CrewAI 1.x forçou downgrade de `chromadb` (1.5 → 1.1) — vale acompanhar. (c) **Surpresa maior**: a `llama3.2:1b` que serve perfeitamente o chat principal **não consegue** fazer function calling do CrewAI — produz JSON malformado e a Crew vira ruído. Subir para `gpt-oss:20b-cloud` (Ollama Cloud free tier) resolveu, mas só depois de **endurecer os backstories** ("SEMPRE chame a tool, NUNCA invente nomes pessoais") porque o modelo grande tem confiança suficiente para "chutar" dados plausíveis se a instrução não fechar a porta para isso.
- **O que aprendi**:
  1. "Métrica justificada para o contexto" foi o exercício mais formativo do projeto inteiro — mais que treinar o modelo. Modelos sem narrativa de negócio são commodities.
  2. **Tamanho de modelo importa de forma não-linear**. Para chat livre, a 1B é suficiente. Para function calling com schemas e tools, há um cliff em torno de 7B abaixo do qual o agente não funciona. Saber onde está esse cliff é parte da maturidade do AI engineer.
  3. **Observabilidade é a diferença entre "a app está estranha" e "olha aqui no trace, o roteador está alucinando".** Sem o Langfuse, eu teria atribuído o bug do roteador a "modelo pequeno é ruim" — com o trace, vi exatamente que o problema era keyword matching em uma palavra ambígua ("guardar").
  4. Backstories de agentes funcionam como **system prompts em escala fina** — a diferença entre "você é um analista financeiro" e "você é um analista financeiro que SEMPRE chama a tool e NUNCA inventa colunas" pode ser a diferença entre demo apresentável e demo embaraçoso.

---

**Total de palavras**: aproximadamente 1.250 (dentro da faixa 700–1500 da rubrica).
