# FIAP - Faculdade de Informática e Administração Paulista

<p align="center">
<img src="https://www.fiap.com.br/wp-content/themes/fiap2016/images/sharing/fiap.png" alt="FIAP Logo" width="200"/>
</p>

# 🌍 OrbitClima — Sistema Inteligente de Monitoramento Climático via Dados Satelitais

**Global Solution 2026.1 — Economia Espacial**

---

## 👥 Integrantes do Grupo

| Nome | RM | Turma |
|------|-----|-------|
| [Nome do Integrante 1] | RM XXXXX | TIAO |
| [Nome do Integrante 2] | RM XXXXX | TIAO |
| [Nome do Integrante 3] | RM XXXXX | TIAO |

> ⚠️ **Atenção:** substitua os campos acima com os nomes e RMs reais do grupo.

---

## 👩‍🏫 Professores Orientadores

- Prof. Caique Sanches — Coordenador GS 2026.1

---

## 📋 Sobre o Projeto

O **OrbitClima** é uma prova de conceito (POC) de sistema inteligente de monitoramento e previsão climática baseado em dados de satélites. A solução integra:

- **Dados orbitais reais** (NASA POWER API — irradiação solar, temperatura, precipitação, ventos)
- **Machine Learning** para detecção de anomalias e previsão de eventos climáticos extremos
- **IA Generativa (LLM)** para geração automática de relatórios em linguagem natural
- **Dashboard interativo** para visualização em tempo real dos indicadores climáticos
- **Alertas automáticos** para autoridades e gestores de risco

A proposta responde à pergunta central da GS 2026.1:

> *"Como tecnologias avançadas de Inteligência Artificial e computação podem impulsionar a nova economia espacial e gerar impacto positivo na Terra?"*

O OrbitClima demonstra como dados gerados por satélites — parte fundamental da economia espacial — podem salvar vidas e apoiar decisões críticas de adaptação climática.

---

## 🛠️ Tecnologias Utilizadas

| Camada | Tecnologia |
|--------|-----------|
| Ingestão de dados | NASA POWER API, Requests, Pandas |
| Processamento | Python 3.11, NumPy, Scikit-learn |
| Modelos preditivos | Isolation Forest (anomalias), LSTM (séries temporais) |
| IA Generativa | LangChain + OpenAI GPT-4o (relatórios automáticos) |
| Visualização | Streamlit, Plotly, Folium (mapas) |
| Infraestrutura | AWS S3 (armazenamento), AWS Lambda (processamento) |
| Embarcado (extensão) | ESP32 + sensores (temperatura, umidade) como dado local |

---

## 📁 Estrutura do Repositório

```
GS-2026-OrbitClima/
│
├── README.md                  ← você está aqui
│
├── src/
│   ├── main.py                ← ponto de entrada da aplicação
│   ├── data_ingestion.py      ← coleta dados da API NASA POWER
│   ├── climate_analyzer.py    ← modelos ML de análise e previsão
│   ├── ai_report_generator.py ← geração de relatórios com IA Generativa
│   ├── dashboard.py           ← dashboard Streamlit interativo
│   └── requirements.txt       ← dependências Python
│
├── docs/
│   ├── arquitetura.md         ← diagrama e descrição da arquitetura
│   └── roteiro_video.md       ← script do vídeo demonstrativo
│
└── assets/
    └── (imagens, diagramas, prints)
```

---

## 🚀 Como Executar

### Pré-requisitos

- Python 3.10+
- Conta OpenAI (para IA Generativa) — opcional para rodar a análise básica
- Conta AWS (para módulo cloud) — opcional para demo local

### Instalação

```bash
# Clonar o repositório
git clone https://github.com/SEU_USUARIO/GS-2026-OrbitClima.git
cd GS-2026-OrbitClima

# Instalar dependências
pip install -r src/requirements.txt

# Configurar variáveis de ambiente
cp .env.example .env
# Editar .env com suas chaves (OPENAI_API_KEY, AWS_ACCESS_KEY etc.)

# Executar análise climática
python src/main.py

# Abrir dashboard (em outro terminal)
streamlit run src/dashboard.py
```

---

## 🌐 Disciplinas Contempladas

Esta solução abrange conteúdos das **Fases 6 e 7** do curso de IA:

- **IA Generativa** — Geração de relatórios automáticos com LLM (GPT-4o + LangChain)
- **Computação em Nuvem** — AWS S3 e Lambda para armazenamento e processamento
- **Análise de Dados** — Pipeline de dados com NASA POWER API
- **Machine Learning** — Isolation Forest e LSTM para previsão e detecção de anomalias
- **Sistemas Embarcados** — ESP32 com sensores como dado local complementar
- **Aplicações Mobile/Web** — Dashboard Streamlit acessível via browser

---

## 📺 Vídeo Demonstrativo

> Link do vídeo: [inserir link do YouTube]

---

## 📄 Licença

Este projeto está licenciado sob a [MIT License](LICENSE).

---

<p align="center">
Desenvolvido com ❤️ para a Global Solution FIAP 2026.1
</p>
