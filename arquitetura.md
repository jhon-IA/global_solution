# 🏗️ Arquitetura da Solução — OrbitClima

## Visão Geral

O **OrbitClima** é uma plataforma de monitoramento climático inteligente que integra
dados de satélites com Machine Learning e IA Generativa para detectar eventos climáticos
extremos e gerar alertas automáticos em linguagem natural.

---

## Diagrama de Arquitetura

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                         🛰️  ORBITCLIMA — ARQUITETURA                         ║
╚══════════════════════════════════════════════════════════════════════════════╝

  ┌─────────────────────────────────────────────────────────────────────────┐
  │                         CAMADA DE DADOS (Entrada)                       │
  │                                                                         │
  │   🛰️ Satélites NASA          📡 Sensor ESP32 Local                      │
  │   (MERRA-2, CERES)           (temperatura, umidade)                     │
  │        │                              │                                  │
  │        ▼                              ▼                                  │
  │   NASA POWER API              MQTT / Serial                              │
  │   (REST, JSON)                (dados embarcados)                         │
  └───────────────────────────────┬─────────────────────────────────────────┘
                                  │
                                  ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │                      CAMADA DE INGESTÃO E ARMAZENAMENTO                 │
  │                                                                         │
  │   [data_ingestion.py]    →    AWS S3 Bucket                             │
  │   • Fetch NASA POWER API      (dados brutos + processados)              │
  │   • Validação e limpeza       (Parquet / JSON)                           │
  │   • Conversão para DataFrame                                             │
  └───────────────────────────────┬─────────────────────────────────────────┘
                                  │
                                  ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │                         CAMADA DE PROCESSAMENTO (IA/ML)                 │
  │                                                                         │
  │   [climate_analyzer.py]                                                 │
  │                                                                         │
  │   ┌─────────────────────┐    ┌──────────────────────────────────┐       │
  │   │  Isolation Forest   │    │  LSTM Bidirecional               │       │
  │   │  (Anomaly Detection)│    │  (Temperature Forecasting)       │       │
  │   │                     │    │                                  │       │
  │   │ • Detecta eventos   │    │ • Lookback: 30 dias              │       │
  │   │   fora do padrão    │    │ • Forecast: 7 dias               │       │
  │   │ • Score por evento  │    │ • MAE < 1.5°C                    │       │
  │   └─────────────────────┘    └──────────────────────────────────┘       │
  │                                                                         │
  │   ┌─────────────────────────────────────────────────────────────┐       │
  │   │  Classificador de Risco (Heurístico + Limiares Científicos) │       │
  │   │  NORMAL → ATENÇÃO → ALTO RISCO → CRÍTICO                    │       │
  │   └─────────────────────────────────────────────────────────────┘       │
  └───────────────────────────────┬─────────────────────────────────────────┘
                                  │
                                  ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │                         CAMADA DE IA GENERATIVA                         │
  │                                                                         │
  │   [ai_report_generator.py]                                              │
  │                                                                         │
  │   LangChain → GPT-4o (OpenAI API)                                       │
  │                                                                         │
  │   • Boletim técnico para autoridades                                    │
  │   • Alerta simplificado para população                                  │
  │   • Análise técnica para cientistas                                     │
  │   • SMS/Push notification automático                                    │
  └───────────────────────────────┬─────────────────────────────────────────┘
                                  │
                                  ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │                         CAMADA DE APRESENTAÇÃO                          │
  │                                                                         │
  │   [dashboard.py] — Streamlit + Plotly + Folium                          │
  │                                                                         │
  │   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌───────────┐  │
  │   │  Mapa        │  │  Séries      │  │  Boletim IA  │  │  KPIs     │  │
  │   │  Interativo  │  │  Temporais   │  │  Gerado em   │  │  Tempo    │  │
  │   │  (Folium)    │  │  (Plotly)    │  │  LN          │  │  Real     │  │
  │   └──────────────┘  └──────────────┘  └──────────────┘  └───────────┘  │
  └─────────────────────────────────────────────────────────────────────────┘
```

---

## Componentes Detalhados

### 1. Ingestão de Dados (data_ingestion.py)

**Fonte:** NASA POWER API (https://power.larc.nasa.gov)

Dados coletados por ponto geográfico (lat/lon):

| Parâmetro | Descrição | Unidade |
|-----------|-----------|---------|
| T2M | Temperatura a 2m de altitude | °C |
| T2M_MAX | Temperatura máxima diária | °C |
| T2M_MIN | Temperatura mínima diária | °C |
| PRECTOTCORR | Precipitação corrigida | mm/dia |
| RH2M | Umidade relativa a 2m | % |
| WS10M | Velocidade do vento a 10m | m/s |
| ALLSKY_SFC_SW_DWN | Irradiação solar global | kWh/m²/dia |

Resolução temporal: diária | Latência: ~7 dias | Cobertura: global

---

### 2. Modelos de Machine Learning (climate_analyzer.py)

#### Modelo A — Isolation Forest (Detecção de Anomalias)
- **Algoritmo:** sklearn.ensemble.IsolationForest
- **Hiperparâmetros:** n_estimators=200, contamination=0.05
- **Input:** Todas as variáveis climáticas normalizadas (MinMaxScaler)
- **Output:** Score de anomalia + flag booleana por dia
- **Justificativa:** Algoritmo não-supervisionado ideal para dados sem rótulos históricos de eventos extremos

#### Modelo B — LSTM Bidirecional (Previsão de Temperatura)
- **Arquitetura:** Bidirectional(LSTM(64)) → Dropout(0.2) → LSTM(32) → Dense(7)
- **Lookback:** 30 dias históricos
- **Forecast:** 7 dias futuros
- **Loss:** Huber (robusta a outliers climáticos)
- **Early stopping:** patience=10 para evitar overfitting

#### Classificador de Risco
Baseado em limiares da Organização Meteorológica Mundial (OMM) e INMET:

| Nível | Temperatura | Precipitação | Vento |
|-------|------------|--------------|-------|
| NORMAL | ≤ 35°C | ≤ 50 mm/d | ≤ 15 m/s |
| ATENÇÃO | 35–40°C | 50–100 mm/d | 15–25 m/s |
| ALTO RISCO | 40–44°C | 100–150 mm/d | — |
| CRÍTICO | > 44°C | > 150 mm/d | > 25 m/s |

---

### 3. IA Generativa (ai_report_generator.py)

**Stack:** LangChain 0.1.x + OpenAI GPT-4o

**Prompt Engineering:**
- System prompt especializado em climatologia e comunicação de risco
- Instruções específicas por público-alvo (autoridades / população / técnico)
- Context window com dados estruturados: estatísticas, anomalias, previsões

**Saídas geradas:**
- Boletim climático completo (~800 palavras)
- Alerta SMS/Push (máx. 160 caracteres)
- Recomendações de ação por nível de risco

**Fallback:** Modo demonstração sem necessidade de API Key (template pré-formatado)

---

### 4. Dashboard (dashboard.py)

**Framework:** Streamlit 1.33

**Componentes visuais:**
- `st.metric()` — KPIs em tempo real
- `plotly.graph_objects.Scatter` — Séries temporais com anomalias destacadas
- `plotly.express.bar` — Precipitação colorizada por nível de risco
- `plotly.express.pie` — Distribuição de dias por risco
- `folium.Map` — Mapa interativo georreferenciado

---

### 5. Extensão IoT — ESP32 (opcional)

**Microcontrolador:** ESP32-WROOM-32

**Sensores conectados:**
- DHT22 — temperatura e umidade local
- BMP280 — pressão atmosférica
- GPS NEO-6M — geolocalização automática

**Comunicação:** MQTT (broker local) → Python (paho-mqtt) → DataFrame

Permite enriquecer os dados satelitais com medições in situ locais.

---

### 6. Infraestrutura Cloud (AWS)

| Serviço | Uso |
|---------|-----|
| AWS S3 | Armazenamento dos dados brutos e processados |
| AWS Lambda | Trigger automático para nova coleta diária |
| AWS CloudWatch | Monitoramento de execução do pipeline |
| AWS SNS | Envio de alertas por e-mail/SMS automático |

---

## Fluxo de Dados Completo

```
NASA POWER API
     │
     │ REST JSON
     ▼
data_ingestion.py ──► Pandas DataFrame (raw)
     │
     │ Dados limpos
     ▼
climate_analyzer.py
     ├── Isolation Forest ──► anomalia (bool) + score
     ├── LSTM ──────────────► forecast (7 dias)
     └── Risk Classifier ──► nivel_risco (4 níveis)
     │
     │ DataFrame enriquecido
     ▼
ai_report_generator.py
     ├── LangChain + GPT-4o ──► Boletim (texto longo)
     └── Template ────────────► Alerta SMS (texto curto)
     │
     │ Resultados
     ▼
dashboard.py (Streamlit)
     ├── Gráficos interativos (Plotly)
     ├── Mapa georreferenciado (Folium)
     └── Boletim + Download
```

---

## Decisões de Design

**Por que NASA POWER API?**
É gratuita, não requer autenticação, tem cobertura global e fornece dados derivados de satélites reais (MERRA-2 e CERES) com validação científica da NASA.

**Por que Isolation Forest?**
Dados climáticos históricos não têm rótulos de "evento extremo". O Isolation Forest é o algoritmo mais robusto para detecção de anomalias não-supervisionada em dados de alta dimensão.

**Por que LSTM Bidirecional?**
Séries temporais climáticas têm dependências temporais complexas. O LSTM bidirecional captura padrões tanto no passado próximo quanto em tendências de longo prazo.

**Por que LangChain + GPT-4o?**
A geração de relatórios técnicos e alertas populacionais com qualidade profissional requer um LLM de alta capacidade. LangChain oferece abstração para troca fácil de modelos e gestão de prompts.
