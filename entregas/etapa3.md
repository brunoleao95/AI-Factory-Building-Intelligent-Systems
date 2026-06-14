# Etapa 3 — Agir: Análise Crítica

> Documento de análise crítica do CBL (fase **Agir**). Cobre as três partes exigidas: revisão do desafio (A), análise técnica (B) e visão de futuro (C). Baseia-se no estado entregue em [`etapa1.md`](etapa1.md) e [`etapa2.md`](etapa2.md), acrescentando a camada de segurança e a publicação.

---

## Parte A — Revisitando o desafio

**Link do site:** https://ai-factory-building-intelligent-systems-mewz2xaperhawiznvrykwv.streamlit.app/ (o chat responde quando o backend local + ngrok estao ativos na maquina da psicologa).

Retomando o que foi definido na Etapa 1:

- **Grande ideia:** gestão inteligente e produtividade tecnológica na psicologia clínica.
- **Pergunta essencial:** como uma psicóloga autônoma pode centralizar gestão financeira e domínio técnico em uma única interface conversacional?
- **Desafio:** construir um assistente que (a) responde sobre o consultório em linguagem natural, (b) consulta material técnico com fontes, e (c) sinaliza padrões de risco financeiro.

**O sistema resolve o desafio.** Os três eixos do desafio foram implementados e funcionam de ponta a ponta, com ressalvas de maturidade próprias de um protótipo acadêmico que roda em LLM local pequeno.

- **Gestão financeira em linguagem natural (a):** atendida. O Text-to-SQL sobre DuckDB responde perguntas de pagamento, faturamento e inadimplência, e o FLAML acrescenta predição de risco de inadimplência (ROC-AUC 0,78). A geração de mensagens de cobrança empáticas automatiza uma tarefa recorrente.
- **Repositório técnico com fontes (b):** atendida. O RAG sobre ChromaDB recupera trechos do DSM-5, código de ética do CFP e técnicas de TCC, citando o arquivo de origem.
- **Sinalização de risco (c):** atendida. O classificador rotula pacientes em risco alto/médio/baixo e alimenta tanto o chat quanto o agente de cobrança.

A medida em que resolve é a de um **protótipo funcional, não um produto de produção**. A interface conversacional única existe e centraliza os três eixos; a qualidade das respostas é limitada pelo `llama3.2:1b` (rápido, porém raso) e o deploy depende da máquina local ligada. O desafio de centralização foi cumprido; o de robustez de produção fica como evolução.

---

## Parte B — Análise técnica

### O que funciona bem

- **Separação em camadas:** `database`, `rag`, `llm`, `ml_model`, `agents`, `crew`, `guardrails` têm interfaces estáveis. Isso permitiu plugar a segurança da Etapa 3 sem reescrever a Etapa 2 — a camada de guardrails apenas envolve `process_question`.
- **Text-to-SQL com few-shot:** os exemplos no prompt ([`src/llm.py`](../src/llm.py)) tornam as queries financeiras típicas confiáveis, com bloqueio de operações destrutivas no DuckDB.
- **AutoML com métrica justificada:** o FLAML escolheu XGBoost; a priorização do **recall** sobre a precisão é coerente com o custo assimétrico do falso negativo no consultório (sessão sem pagamento não mitigável).
- **Guardrails em profundidade:** a combinação regex (PT+EN) + modelo HF de injection + LLM-juiz + Presidio bloqueia injection, jailbreak, toxicidade e orientação clínica indevida, e anonimiza PII (incluindo CPF e telefone pt-BR). A demonstração ([`tests/test_guardrails.py`](../tests/test_guardrails.py)) bloqueia 5 ataques e libera as perguntas legítimas.

### O que funciona mal e por quê

- **Roteador por keywords é frágil:** `route_question` ([`src/agents.py`](../src/agents.py)) classifica por contagem de palavras-chave. Palavras ambíguas como "guardar" e "tempo" foram contadas como sinal financeiro e enviaram uma pergunta de RAG ("por quanto tempo guardar prontuários?") ao Text-to-SQL, gerando SQL inválido (`Binder Error`). A causa é a ausência de compreensão semântica no roteamento.
- **Modelo de injection é treinado em inglês:** o `protectai/deberta-v3-base-prompt-injection-v2` classifica "Quais pacientes estão pendentes?" (legítima) como `INJECTION` com score 0,999, enquanto acerta "Quanto recebi em outubro?". O modelo é confiável em inglês e em ataques, mas gera falso-positivo em português legítimo. A mitigação foi usar o LLM-juiz em PT como desempate: o modelo só bloqueia se o juiz confirmar, o que protege o usuário legítimo.
- **Juiz `llama3.2:1b` é instável em prompts estruturados:** na avaliação DeepEval da Etapa 2, o juiz produziu justificativas incoerentes em itens com schema JSON longo. O modelo é forte em conversa direta e fraco em raciocínio estruturado.
- **Latência:** o `llama3.2:1b` responde em segundos por turno; a Crew (`gpt-oss:20b-cloud`) é mais lenta ainda. Aceitável para demonstração, ruim para uso contínuo.

### Limitações conhecidas

- **Tecnologia:** LLM local pequeno (qualidade e latência limitadas); cold start do modelo HF de injection na primeira execução; sanitização de PII na saída atua sobre o texto completo (não token-a-token) por causa do streaming.
- **Dados:** pacientes e financeiro são sintéticos (seed fixa); o modelo de risco foi treinado em dataset público (UCI) mapeado ao domínio, não em dados reais do consultório.
- **Escopo:** o assistente é de gestão; não oferece orientação clínica (bloqueada por design).
- **Deploy:** a arquitetura híbrida (frontend no Streamlit Cloud + backend local via ngrok) entrega URL pública, mas o chat depende da máquina da psicóloga ligada.

### Ética e transparência

- **Aviso de IA:** a interface declara que o usuário fala com uma IA e que respostas podem conter erros, não substituindo julgamento profissional.
- **Privacidade:** PII inserida no chat (nome, CPF, telefone, e-mail) é anonimizada antes de chegar ao LLM e aos logs; os pacientes já são identificados por código (PAC-ALPHA) por sigilo.
- **Viés:** LLMs reproduzem vieses dos dados de treinamento e têm desempenho melhor em inglês — observado de forma concreta no falso-positivo do modelo de injection em português, o que reforça a necessidade da camada de desempate em PT.

---

## Parte C — Visão de futuro

**1. Substituir o roteador por keywords por um classificador LLM-as-router.**
- **O que muda:** trocar a contagem de palavras-chave por uma chamada de classificação (cinco categorias) com escolha por maior probabilidade.
- **Tecnologia:** um LLM pequeno dedicado ao roteamento ou um classificador de intenção fine-tunado.
- **Impacto:** elimina erros como "guardar/tempo" enviados ao Text-to-SQL, aumentando a taxa de respostas corretas, especialmente em perguntas de RAG.

**2. Tornar o backend sempre disponível com LLM hospedado.**
- **O que muda:** mover a inferência do Ollama local para um endpoint hospedado e publicar o backend numa plataforma always-on.
- **Tecnologia:** Ollama Cloud ou Groq para o LLM; Hugging Face Spaces (Docker) para o backend, que suporta as dependências pesadas.
- **Impacto:** remove a dependência da máquina ligada — a URL responde 24/7, requisito de um produto real.

**3. Elevar a qualidade do modelo e da avaliação.**
- **O que muda:** usar um LLM maior como juiz e ajustar o limiar de decisão do FLAML para subir o recall.
- **Tecnologia:** `llama3.1:8b` (local) como juiz dedicado; threshold de risco em 0,30 em vez de 0,50; golden dataset expandido de 15 para 30 perguntas.
- **Impacto:** avaliação mais consistente e captura de mais inadimplentes (recall maior), ao custo de mais falsos positivos toleráveis.

**4. Endurecer os guardrails com modelos multilíngues dedicados.**
- **O que muda:** adicionar um modelo de toxicidade e um zero-shot de tópicos treinados/avaliados em português, complementando a camada atual.
- **Tecnologia:** modelos XLM-R multilíngues; opcionalmente Presidio com mais reconhecedores pt-BR (RG, CNS).
- **Impacto:** menos dependência do regex e do juiz, com detecção mais robusta em português.

---

## Apoio ao vídeo

> Roteiro sugerido — substituir pelas suas palavras e experiência real.

1. **Mostrar o site no ar:** abrir a URL pública e fazer uma pergunta financeira, uma de RAG e um ataque sendo bloqueado.
2. **O que aprendi que não esperava:** _preencher (ex.: o quanto o tamanho do modelo importa de forma não-linear para function calling; o quanto observabilidade muda o diagnóstico de bugs)._
3. **Conexão com objetivos profissionais/pessoais:** _preencher (motivação pessoal já descrita no README)._
4. **O que faria diferente:** _preencher (ex.: começar pelo roteador semântico; escolher Python 3.11 para compatibilidade com mais libs de segurança)._
