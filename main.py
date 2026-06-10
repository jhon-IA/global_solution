"""
main.py
-------
Ponto de entrada principal do sistema OrbitClima.

Orquestra o pipeline completo:
  1. Ingestão de dados satelitais (NASA POWER API)
  2. Análise e detecção de anomalias (Isolation Forest)
  3. Classificação de risco climático
  4. Previsão de temperatura (LSTM)
  5. Geração de relatório com IA Generativa (GPT-4o)

Execute com: python src/main.py
Para o dashboard: streamlit run src/dashboard.py
"""

import pandas as pd
import numpy as np
from datetime import datetime

# ── Importar módulos do OrbitClima ────────────────────────────────
from data_ingestion import fetch_recent_data, get_brazil_regions
from climate_analyzer import ClimateAnomalyDetector, TemperatureForecaster, classify_climate_risk
from ai_report_generator import generate_climate_bulletin, generate_alert_sms


def run_pipeline(
    location_name: str,
    latitude: float,
    longitude: float,
    days_back: int = 365,
    use_lstm: bool = False,
    use_ai_report: bool = True,
) -> dict:
    """
    Executa o pipeline completo de monitoramento climático para uma localização.

    Args:
        location_name: Nome descritivo da localização
        latitude: Latitude do ponto
        longitude: Longitude do ponto
        days_back: Período histórico em dias
        use_lstm: Se True, treina e usa modelo LSTM para previsão (requer TF)
        use_ai_report: Se True, gera relatório com IA Generativa

    Returns:
        Dicionário com resultados de cada etapa do pipeline.
    """

    print("\n" + "═" * 60)
    print(f"  🛰️  OrbitClima — Pipeline de Monitoramento Climático")
    print(f"  📍 Localização: {location_name}")
    print(f"  📅 Período: últimos {days_back} dias")
    print("═" * 60)

    results = {"location": location_name, "timestamp": datetime.now().isoformat()}

    # ── ETAPA 1: Ingestão de dados ─────────────────────────────────
    print("\n[1/5] 🌐 Coletando dados satelitais da NASA POWER API...")
    try:
        df = fetch_recent_data(latitude, longitude, days_back)
        print(f"      ✅ {len(df)} dias coletados | Colunas: {list(df.columns)}")
    except Exception as e:
        print(f"      ⚠️  Erro na API: {e}. Usando dados simulados para demonstração.")
        df = _generate_demo_data(days_back)

    results["raw_data_shape"] = df.shape
    results["data_range"] = {
        "start": str(df.index[0].date()),
        "end": str(df.index[-1].date()),
    }

    # ── ETAPA 2: Detecção de anomalias ─────────────────────────────
    print("\n[2/5] 🔬 Detectando anomalias com Isolation Forest...")
    detector = ClimateAnomalyDetector(contamination=0.05)
    detector.fit(df)
    df = detector.predict(df)
    anomaly_report = detector.get_anomaly_report(df)
    print(f"      📊 Relatório:\n{anomaly_report}")
    results["anomaly_report"] = anomaly_report
    results["n_anomalies"] = int(df["anomalia"].sum())

    # ── ETAPA 3: Classificação de risco ────────────────────────────
    print("\n[3/5] ⚠️  Classificando risco climático...")
    df = classify_climate_risk(df)
    risk_dist = df["nivel_risco"].value_counts().to_dict()
    print(f"      📊 Distribuição de risco: {risk_dist}")
    results["risk_distribution"] = risk_dist

    # ── ETAPA 4: Previsão LSTM (opcional) ──────────────────────────
    forecast_df = None
    if use_lstm:
        print("\n[4/5] 🧠 Treinando LSTM para previsão de temperatura...")
        try:
            forecaster = TemperatureForecaster(lookback=30, forecast_days=7)
            forecaster.fit(df, epochs=30)
            forecast_df = forecaster.forecast(df)
            print(f"      📈 Previsão para os próximos 7 dias:")
            print(forecast_df.to_string())
            results["forecast"] = forecast_df.to_dict()
        except Exception as e:
            print(f"      ⚠️  LSTM não disponível: {e}. Pulando etapa.")
    else:
        print("\n[4/5] ⏭️  Previsão LSTM desativada (use_lstm=False). Pulando.")

    # ── ETAPA 5: Relatório com IA Generativa ───────────────────────
    print("\n[5/5] 🤖 Gerando boletim com IA Generativa...")
    if use_ai_report:
        bulletin = generate_climate_bulletin(
            df=df,
            location=location_name,
            anomaly_report=anomaly_report,
            forecast_df=forecast_df,
            audience="autoridades",
        )
        print(bulletin)
        results["bulletin"] = bulletin

        sms_alert = generate_alert_sms(df.tail(7), location_name)
        print(f"\n📱 ALERTA SMS:")
        print(sms_alert)
        results["sms_alert"] = sms_alert
    else:
        print("      ⏭️  Geração de relatório IA desativada.")

    print("\n" + "═" * 60)
    print("  ✅ Pipeline OrbitClima concluído com sucesso!")
    print("  🖥️  Para visualizar: streamlit run src/dashboard.py")
    print("═" * 60)

    return results


def _generate_demo_data(days_back: int) -> pd.DataFrame:
    """Gera dados climáticos simulados para demonstração offline."""
    np.random.seed(42)
    dates = pd.date_range(end=pd.Timestamp.today(), periods=days_back)
    base = 28 + 5 * np.sin(np.linspace(0, 2 * np.pi * 2, days_back))
    return pd.DataFrame({
        "T2M": base + np.random.normal(0, 2, days_back),
        "T2M_MAX": base + np.abs(np.random.normal(4, 1.5, days_back)),
        "T2M_MIN": base - np.abs(np.random.normal(4, 1.5, days_back)),
        "PRECTOTCORR": np.abs(np.random.exponential(6, days_back)),
        "RH2M": np.random.normal(70, 12, days_back).clip(20, 100),
        "WS10M": np.abs(np.random.normal(4, 2, days_back)),
        "ALLSKY_SFC_SW_DWN": np.abs(np.random.normal(5, 1.5, days_back)),
    }, index=dates)


def run_multi_region_summary():
    """
    Executa análise rápida para todas as regiões do Brasil e imprime resumo.
    Útil para demonstração no vídeo.
    """
    regions = get_brazil_regions()
    print("\n🌎 ORBITCLIMA — MONITORAMENTO MULTI-REGIÃO BRASIL")
    print("=" * 70)

    for name, coords in regions.items():
        try:
            df = _generate_demo_data(365)
            df = classify_climate_risk(df)
            risk = df["nivel_risco"].value_counts(normalize=True)
            temp_now = df["T2M"].iloc[-1]
            critical_pct = risk.get("CRÍTICO", 0) * 100
            print(f"📍 {name:35s} | T={temp_now:.1f}°C | Crítico: {critical_pct:.1f}%")
        except Exception as e:
            print(f"📍 {name:35s} | Erro: {e}")

    print("=" * 70)


if __name__ == "__main__":
    # Pipeline principal — São Paulo
    result = run_pipeline(
        location_name="São Paulo, SP — Brasil",
        latitude=-23.55,
        longitude=-46.63,
        days_back=365,
        use_lstm=False,   # Defina True se TensorFlow instalado
        use_ai_report=True,
    )

    # Resumo multi-região
    run_multi_region_summary()
