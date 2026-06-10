"""
dashboard.py
------------
Dashboard interativo OrbitClima construído com Streamlit + Plotly.

Para executar: streamlit run src/dashboard.py

O dashboard exibe:
- Mapa interativo com regiões monitoradas
- Série temporal de temperatura e precipitação
- Eventos anômalos destacados
- Previsões futuras do modelo LSTM
- Boletim climático gerado por IA
- Alertas em tempo real por nível de risco
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# ─── Configuração da página ──────────────────────────────────────
st.set_page_config(
    page_title="OrbitClima 🛰️",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS personalizado ───────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1a73e8;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #5f6368;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e3c72, #2a5298);
        border-radius: 12px;
        padding: 1rem;
        color: white;
        text-align: center;
    }
    .risk-critical { background-color: #d32f2f; color: white; border-radius: 8px; padding: 0.5rem; }
    .risk-high { background-color: #f57c00; color: white; border-radius: 8px; padding: 0.5rem; }
    .risk-attention { background-color: #f9a825; border-radius: 8px; padding: 0.5rem; }
    .risk-normal { background-color: #388e3c; color: white; border-radius: 8px; padding: 0.5rem; }
</style>
""", unsafe_allow_html=True)


# ─── Header ──────────────────────────────────────────────────────
st.markdown('<p class="main-header">🛰️ OrbitClima</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Sistema Inteligente de Monitoramento Climático via Dados Satelitais — NASA POWER API</p>',
    unsafe_allow_html=True,
)
st.divider()


# ─── Sidebar ─────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://www.nasa.gov/wp-content/themes/nasa/assets/images/nasa-logo.svg", width=100)
    st.markdown("### ⚙️ Configurações")

    regioes = {
        "São Paulo (Metrópole)": (-23.55, -46.63),
        "Amazônia (Manaus)": (-3.10, -60.02),
        "Pantanal (Corumbá)": (-19.01, -57.65),
        "Nordeste Semiárido (Petrolina)": (-9.38, -40.51),
        "Sul (Porto Alegre)": (-30.03, -51.23),
        "Cerrado (Brasília)": (-15.78, -47.93),
    }

    selected_region = st.selectbox("📍 Região monitorada", list(regioes.keys()))
    lat, lon = regioes[selected_region]

    days_back = st.slider("📅 Período de análise (dias)", min_value=90, max_value=730, value=365, step=30)
    contamination = st.slider("🔬 Sensibilidade de anomalias", 0.01, 0.15, 0.05, 0.01,
                              help="Proporção de eventos considerados anômalos")

    use_real_api = st.toggle("🌐 Usar NASA POWER API (real)", value=False,
                             help="Desative para usar dados simulados (demo rápida)")

    st.markdown("---")
    st.markdown("**📡 Fonte dos dados:**")
    st.markdown("NASA POWER API\n(satélites MERRA-2 e CERES)")
    st.markdown("**🤖 IA:** GPT-4o via LangChain")


# ─── Geração / carregamento de dados ─────────────────────────────

@st.cache_data(ttl=3600)
def load_data(lat, lon, days_back, use_real):
    if use_real:
        try:
            from data_ingestion import fetch_recent_data
            return fetch_recent_data(lat, lon, days_back)
        except Exception as e:
            st.warning(f"Erro na API NASA: {e}. Usando dados simulados.")

    # Dados simulados para demonstração
    np.random.seed(int(abs(lat * 100)))
    dates = pd.date_range(end=pd.Timestamp.today(), periods=days_back)
    base_temp = 25 + 5 * np.sin(np.linspace(0, 2 * np.pi, days_back)) + np.random.normal(0, 2, days_back)

    df = pd.DataFrame({
        "T2M": base_temp,
        "T2M_MAX": base_temp + np.abs(np.random.normal(4, 1.5, days_back)),
        "T2M_MIN": base_temp - np.abs(np.random.normal(4, 1.5, days_back)),
        "PRECTOTCORR": np.abs(np.random.exponential(6, days_back)),
        "RH2M": np.random.normal(70, 12, days_back).clip(20, 100),
        "WS10M": np.abs(np.random.normal(4, 2, days_back)),
        "ALLSKY_SFC_SW_DWN": np.abs(np.random.normal(5, 1.5, days_back)),
    }, index=dates)

    # Injetar eventos extremos simulados
    df.iloc[30, 0] = 43.5; df.iloc[30, 1] = 48.0   # onda de calor
    df.iloc[90, 2] = 160.0                            # chuva extrema
    df.iloc[180, 0] = 44.2                            # segundo evento de calor

    return df

with st.spinner("🛰️ Coletando dados satelitais..."):
    df = load_data(lat, lon, days_back, use_real_api)


# ─── Análise ML ──────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def run_analysis(df_json, contamination):
    df = pd.read_json(df_json)
    df.index = pd.to_datetime(df.index)

    from climate_analyzer import ClimateAnomalyDetector, classify_climate_risk
    detector = ClimateAnomalyDetector(contamination=contamination)
    detector.fit(df)
    df = detector.predict(df)
    df = classify_climate_risk(df)
    anomaly_report = detector.get_anomaly_report(df)
    return df, anomaly_report

df_analyzed, anomaly_report = run_analysis(df.to_json(), contamination)


# ─── KPIs ────────────────────────────────────────────────────────
st.markdown("### 📊 Indicadores Atuais")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("🌡️ Temp. Atual", f"{df_analyzed['T2M'].iloc[-1]:.1f}°C",
              delta=f"{df_analyzed['T2M'].iloc[-1] - df_analyzed['T2M'].mean():.1f}°C vs média")
with col2:
    st.metric("🌧️ Precipitação 7d", f"{df_analyzed['PRECTOTCORR'].tail(7).sum():.0f} mm")
with col3:
    st.metric("💧 Umidade", f"{df_analyzed['RH2M'].iloc[-1]:.0f}%")
with col4:
    st.metric("💨 Vento", f"{df_analyzed['WS10M'].iloc[-1]:.1f} m/s")
with col5:
    n_anom = df_analyzed["anomalia"].sum()
    st.metric("🚨 Anomalias", f"{n_anom}",
              delta=f"{n_anom/len(df_analyzed)*100:.1f}% do período",
              delta_color="inverse")

st.divider()


# ─── Gráficos ────────────────────────────────────────────────────
col_left, col_right = st.columns([2, 1])

with col_left:
    st.markdown("### 🌡️ Temperatura ao Longo do Tempo")

    fig_temp = go.Figure()
    fig_temp.add_trace(go.Scatter(
        x=df_analyzed.index, y=df_analyzed["T2M"],
        name="Temperatura (°C)", line=dict(color="#1a73e8", width=1.5),
    ))
    fig_temp.add_trace(go.Scatter(
        x=df_analyzed.index, y=df_analyzed["T2M_MAX"],
        name="Máxima", line=dict(color="#ea4335", width=1, dash="dot"),
    ))
    fig_temp.add_trace(go.Scatter(
        x=df_analyzed.index, y=df_analyzed["T2M_MIN"],
        name="Mínima", line=dict(color="#34a853", width=1, dash="dot"),
    ))

    # Destacar anomalias
    anomalias = df_analyzed[df_analyzed["anomalia"]]
    fig_temp.add_trace(go.Scatter(
        x=anomalias.index, y=anomalias["T2M"],
        mode="markers", name="⚠️ Anomalia",
        marker=dict(color="red", size=8, symbol="x"),
    ))

    fig_temp.update_layout(
        xaxis_title="Data", yaxis_title="Temperatura (°C)",
        legend=dict(orientation="h", y=-0.2),
        height=350, plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_temp, use_container_width=True)

    st.markdown("### 🌧️ Precipitação Diária")
    risk_colors = {
        "NORMAL": "#34a853", "ATENÇÃO": "#fbbc04",
        "ALTO RISCO": "#fa7b17", "CRÍTICO": "#ea4335",
    }
    fig_prec = px.bar(
        df_analyzed.reset_index(), x="index", y="PRECTOTCORR",
        color="nivel_risco",
        color_discrete_map=risk_colors,
        labels={"index": "Data", "PRECTOTCORR": "Precipitação (mm/dia)", "nivel_risco": "Risco"},
        height=300,
    )
    fig_prec.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_prec, use_container_width=True)

with col_right:
    st.markdown("### 🗺️ Mapa de Monitoramento")
    import folium
    from streamlit_folium import st_folium

    m = folium.Map(location=[lat, lon], zoom_start=6, tiles="CartoDB dark_matter")
    folium.CircleMarker(
        location=[lat, lon], radius=15,
        color="#1a73e8", fill=True, fill_color="#1a73e8", fill_opacity=0.6,
        popup=f"🛰️ {selected_region}\nTemp: {df_analyzed['T2M'].iloc[-1]:.1f}°C",
        tooltip=selected_region,
    ).add_to(m)
    st_folium(m, height=280, width=None)

    st.markdown("### 📊 Distribuição de Risco")
    risk_counts = df_analyzed["nivel_risco"].value_counts()
    fig_pie = px.pie(
        values=risk_counts.values, names=risk_counts.index,
        color=risk_counts.index,
        color_discrete_map=risk_colors,
        hole=0.4,
    )
    fig_pie.update_layout(height=250, paper_bgcolor="rgba(0,0,0,0)", margin=dict(t=0, b=0))
    st.plotly_chart(fig_pie, use_container_width=True)


# ─── Relatório IA ────────────────────────────────────────────────
st.divider()
st.markdown("### 🤖 Boletim Climático Gerado por IA")

if st.button("🔄 Gerar Boletim com IA Generativa", type="primary"):
    with st.spinner("🤖 Gerando boletim com GPT-4o via LangChain..."):
        from ai_report_generator import generate_climate_bulletin
        audience = st.session_state.get("audience", "autoridades")
        bulletin = generate_climate_bulletin(
            df=df_analyzed,
            location=selected_region,
            anomaly_report=anomaly_report,
            audience=audience,
        )
        st.session_state["bulletin"] = bulletin

audience_option = st.selectbox(
    "Público-alvo do boletim:",
    ["autoridades", "populacao", "tecnico"],
    key="audience"
)

if "bulletin" in st.session_state:
    st.code(st.session_state["bulletin"], language=None)
    st.download_button(
        "📥 Baixar Boletim",
        data=st.session_state["bulletin"],
        file_name=f"orbitclima_boletim_{datetime.today().strftime('%Y%m%d')}.txt",
        mime="text/plain",
    )
else:
    st.info("Clique no botão acima para gerar o boletim com IA Generativa.")


# ─── Footer ──────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align:center; color:#9e9e9e; font-size:0.8rem;'>"
    "🛰️ OrbitClima — Global Solution FIAP 2026.1 | "
    "Dados: NASA POWER API | IA: GPT-4o via LangChain | "
    "Desenvolvido com Python, Streamlit e ❤️"
    "</div>",
    unsafe_allow_html=True,
)
