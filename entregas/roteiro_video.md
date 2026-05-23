# Roteiro do Video da Etapa 2 - Demo (4 min 30 s)

> Estrutura cronometrada com falas sugeridas e indicacoes visuais. Pratique 1-2x antes de gravar — o tempo aperta. Manter ritmo natural; melhor cortar conteudo do que correr.

## Pre-gravacao (checklist 5 min antes)

- [ ] Ollama rodando (`ollama serve`)
- [ ] App ja iniciado em `http://localhost:8501` (com Langfuse habilitado)
- [ ] `cloud.langfuse.com` aberto em outra aba, ja logado, no projeto certo
- [ ] `evals/results.md` aberto em editor (VS Code)
- [ ] `models/metrics.json` aberto em editor
- [ ] Terminal limpo, na pasta do projeto
- [ ] Janelas do navegador organizadas em ordem: app -> langfuse -> editor
- [ ] Microfone testado, camera (opcional) testada

---

## 0:00 - 0:20 | Abertura (20s)

**Mostrar na tela**: rosto/camera (opcional) + slide simples ou tela do app na home

**Fala**:
> "Ola, sou o Bruno, aluno de IA Factory. Vou apresentar a etapa 2 do meu projeto: um assistente inteligente para psicologas autonomas que centraliza gestao financeira, busca em documentos tecnicos e analise de risco de inadimplencia. Nessa etapa adicionei quatro camadas: AutoML, agentes CrewAI, observabilidade com Langfuse e avaliacao com DeepEval."

---

## 0:20 - 1:05 | Camada 1: ML com FLAML (45s)

**Mostrar na tela**:
1. Editor com `models/metrics.json` aberto
2. Streamlit: digitar a pergunta `Qual paciente tem maior risco?`

**Fala enquanto mostra o JSON**:
> "Para o ML, usei o FLAML como AutoML sobre o dataset publico do UCI: 'Default of Credit Card Clients', com 30 mil registros. O FLAML escolheu o XGBoost. As metricas no test set foram: accuracy 0,82, ROC-AUC 0,78, recall 0,36."

**Fala enquanto vai para o app**:
> "A metrica que priorizo no contexto da psicologa e o **recall** — porque um falso negativo, ou seja, deixar passar um paciente que vai inadimplir, gera perda financeira real, enquanto um falso positivo so dispara uma mensagem de cobranca extra."

**Fala apontando para a resposta do app**:
> "O modelo esta integrado ao chat. Quando pergunto sobre risco, o sistema chama o `predict_risk` direto na tool e mostra o paciente com maior probabilidade — aqui o PAC-EPSILON com 60%."

---

## 1:05 - 1:45 | Camada 2: Agentes CrewAI (40s)

**Mostrar na tela**: app Streamlit, digitar e enviar:
```
/equipe quem esta inadimplente e gere as mensagens de cobranca
```

**Fala enquanto a Crew executa (~30s)**:
> "Para tarefas compostas uso CrewAI com tres agentes: Analista Financeiro com tools de SQL, Analista de Risco com tools de ML e mensagem de cobranca, e um Editor que consolida o resultado. O prefixo `/equipe` desvia do roteador rapido da etapa 1 e aciona a Crew. Para perguntas simples, o roteador antigo continua respondendo direto."

**Quando a resposta aparecer, apontar para a tabela** (~10s):
> "Os codigos vem reais do banco — PAC-MU, PAC-EPSILON, com valores corretos — porque os agentes invocam as tools de fato. Volto ao processo desta camada na reflexao final, porque o caminho ate aqui foi cheio de iteracao."

---

## 1:45 - 2:30 | Camada 3: Observabilidade Langfuse (45s)

**Mostrar na tela**: alternar para a aba do `cloud.langfuse.com`, ja logado

**Fala mostrando a lista de traces** (Screenshot_1):
> "Toda chamada do pipeline esta instrumentada. Aqui no Langfuse vejo cada `process_question` como um trace, com sub-spans aninhados: `rag_retrieval`, `text_to_sql`, `ollama_chat_stream`, e o `crew_run` quando uso a equipe."

**Clicar em um trace `ollama_chat_stream`** (mostrar tela tipo Screenshot_3):
> "Clicando em um trace de chamada LLM, vejo a latencia exata — 3,25 segundos aqui — o input com as mensagens completas, o output formatado, e os tokens consumidos. E o que a rubrica pede: latencia, tokens e custo visiveis."

**Fala**:
> "Se as chaves do Langfuse nao estiverem no `.env`, o decorator vira no-op e o app continua funcionando local sem dependencia externa."

---

## 2:30 - 3:10 | Camada 4: Golden Dataset + DeepEval (40s)

**Mostrar na tela**: `evals/results.md` aberto no editor

**Fala mostrando o cabecalho**:
> "Para garantir qualidade, montei um golden dataset com 15 perguntas — quatro financeiras, quatro de RAG, tres de risco, duas de cobranca e duas fora-de-escopo. Roda via DeepEval, com a propria llama3.2 como juiz local."

**Mostrar a tabela de scores**:
> "O resultado: faithfulness media 0,91, answer relevancy media 0,74, com 15 de 15 aprovadas no limiar 0,5."

**Mostrar a secao de pontos fracos no relatorio etapa2.md ou no results.md**:
> "Mais valioso que os scores foram os achados qualitativos: descobri um bug do roteador da etapa 1 que mandava perguntas RAG sobre prontuarios para o agente financeiro, gerando SQL invalido. Esse bug aparece tanto no DeepEval quanto num trace alucinado que vi no Langfuse. E vou enderecar isso na etapa 3."

---

## 3:10 - 4:15 | Reflexao pessoal: a saga do CrewAI (1min 5s)

> Esta e a parte que **mais pesa na nota da rubrica** — e a oportunidade de mostrar engenharia real, nao so checklist. A narrativa das 3 tentativas com a Crew e o ouro aqui.

**Mostrar na tela**: camera (opcional) OU app com o resultado da Crew ainda visivel

**Fala — abertura**:
> "Quero contar como foi a iteracao da camada de agentes, que foi a parte mais formativa do projeto."

**Tentativa 1 — modelo pequeno (15s)**:
> "Comecei com a llama3.2 de 1 bilhao de parametros, o mesmo modelo do resto do app. Quando rodei a Crew, em vez de chamar as tools, o modelo gerava texto descrevendo o formato de tool call — repetindo `function name parameters` em loop. Aprendi na pratica que function calling exige modelos a partir de 7 bilhoes de parametros; abaixo disso, o agente nao funciona."

**Tentativa 2 — modelo grande, prompts soltos (15s)**:
> "Troquei para o `gpt-oss:20b-cloud`, no tier gratuito do Ollama Cloud. Acelerou a Crew de 218 segundos para 30, mas apareceu outro problema: o modelo grande tem confianca suficiente para inventar dados plausiveis. Comecou a retornar pacientes chamados Ana Silva, Joao Pereira, com colunas que nao existem no banco, tipo email e telefone. O modelo era capaz, mas as tools nao estavam sendo invocadas."

**Tentativa 3 — backstories restritivos + fallback (20s)**:
> "Endurecei os backstories dos agentes com regras explicitas: 'SEMPRE chame a tool query_finance, NUNCA invente nomes pessoais, codigos sao PAC-ALPHA gregos'. Funcionou — voltou a usar dados reais. Mas surgiu mais um problema: o agente as vezes terminava uma task com tool_call sem texto, e o CrewAI quebrava com um erro de validacao Pydantic. Resolvi tirando todas as tools do agente consolidador (ele so formata) e adicionando fallback: se a Crew falhar, o sistema cai no pipeline simples da etapa 1."

**Fechamento — takeaways (15s)**:
> "Tres aprendizados concretos: primeiro, tamanho de modelo importa de forma nao-linear — tem um cliff de capacidade em torno de 7B parametros. Segundo, backstories de agente sao **system prompts em escala fina** — a diferenca entre demo embaracoso e demo apresentavel. Terceiro, observabilidade do Langfuse foi essencial pra eu ver, em cada tentativa, exatamente o que o agente estava fazendo de errado."

---

## 4:15 - 4:30 | Encerramento (15s)

**Mostrar na tela**: app na tela inicial ou camera

**Fala**:
> "Codigo, dados e relatorio tecnico estao no repositorio. Para a etapa 3 vou focar em corrigir o roteador, fazer deploy do app e adicionar autenticacao. Obrigado!"

---

## Tempos resumidos

| Bloco | Tempo | Acumulado |
|-------|-------|-----------|
| Abertura | 0:20 | 0:20 |
| ML / FLAML | 0:45 | 1:05 |
| CrewAI (demo curto) | 0:40 | 1:45 |
| Langfuse | 0:45 | 2:30 |
| DeepEval | 0:40 | 3:10 |
| Reflexao (saga CrewAI) | 1:05 | 4:15 |
| Encerramento | 0:15 | 4:30 |

**Total: 4 min 30 s**. A reflexao ficou maior que o normal porque e a parte de maior valor narrativo — vale priorizar.

---

## Dicas finais

- **Velocidade da Crew**: agora a `/equipe` leva ~30s com `gpt-oss:20b-cloud`. Da para esperar inteira sem cortar. Se algum dia voltar a demorar, lancar a query no inicio do bloco e narrar enquanto roda.
- **Tela cheia**: gravar com Streamlit em janela maximizada (F11) para nao mostrar barras de browser.
- **Voz**: a rubrica pede que voce **fale** o video. Camera ligada e desejavel mas opcional.
- **Edicao**: se gravar de uma vez ficar dificil, gravar bloco por bloco e juntar — corte total fica mais limpo.
- **Backup**: `evals/results.md` ja gerado, modelo FLAML ja treinado, traces ja no Langfuse. Tudo pronto, nao precisa rodar nada novo durante a gravacao.
- **Para a saga do CrewAI ficar forte**: ensaie a transicao "tentativa 1 -> 2 -> 3" — cronologia clara torna a narrativa convincente. Se gravar bloco-a-bloco, gravar a reflexao por ultimo (e a parte mais densa, vale fazer por ultimo quando ja esta confortavel com o cenario).
