"""
ai_report_generator.py
-----------------------
Módulo de IA Generativa para geração automática de relatórios climáticos
em linguagem natural, utilizando LangChain + OpenAI GPT-4o.

A IA recebe dados estruturados do análise climática e produz:
- Boletins climáticos para autoridades
- Alertas em linguagem simples para a população
- Análises técnicas para gestores de risco
- Recomendações de ação para cada nível de risco

Disciplinas contempladas: IA Generativa, NLP, LangChain, OpenAI API
"""

import os
import json
import pandas as pd
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


def _format_data_for_prompt(
    df: pd.DataFrame,
    location: str,
    anomaly_report: str,
    forecast_df: Optional[pd.DataFrame] = None,
) -> str:
    """
    Formata os dados climáticos em um contexto estruturado para o LLM.
    """
    recent = df.tail(30)

    context = f"""
LOCALIZAÇÃO: {location}
PERÍODO DE ANÁLISE: {df.index[0].strftime('%d/%m/%Y')} a {df.index[-1].strftime('%d/%m/%Y')}
TOTAL DE DIAS ANALISADOS: {len(df)}

ESTATÍSTICAS RECENTES (últimos 30 dias):
- Temperatura média: {recent['T2M'].mean():.1f}°C
- Temperatura máxima registrada: {recent['T2M_MAX'].mean():.1f}°C (máx. histórica: {df['T2M_MAX'].max():.1f}°C)
- Temperatura mínima: {recent['T2M_MIN'].mean():.1f}°C
- Precipitação acumulada: {recent['PRECTOTCORR'].sum():.1f} mm
- Umidade relativa média: {recent['RH2M'].mean():.1f}%
- Velocidade média do vento: {recent['WS10M'].mean():.1f} m/s
- Irradiação solar média: {recent.get('ALLSKY_SFC_SW_DWN', pd.Series([0])).mean():.2f} kWh/m²/dia

RELATÓRIO DE ANOMALIAS:
{anomaly_report}

DISTRIBUIÇÃO DE RISCO:
{df['nivel_risco'].value_counts().to_string() if 'nivel_risco' in df.columns else 'Não calculado'}
"""

    if forecast_df is not None:
        context += f"""
PREVISÃO PARA OS PRÓXIMOS {len(forecast_df)} DIAS (modelo LSTM):
{forecast_df.to_string()}
"""

    return context


def generate_climate_bulletin(
    df: pd.DataFrame,
    location: str,
    anomaly_report: str,
    forecast_df: Optional[pd.DataFrame] = None,
    audience: str = "autoridades",
) -> str:
    """
    Gera boletim climático com IA Generativa (LangChain + GPT-4o).

    Args:
        df: DataFrame com dados climáticos analisados
        location: Nome da localização monitorada
        anomaly_report: Texto do relatório de anomalias
        forecast_df: DataFrame com previsões (opcional)
        audience: Público-alvo — 'autoridades', 'populacao', 'tecnico'

    Returns:
        Boletim climático em linguagem natural.
    """
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        # Modo demo sem API Key
        return _generate_demo_bulletin(df, location, anomaly_report, audience)

    try:
        from langchain_openai import ChatOpenAI
        from langchain.prompts import ChatPromptTemplate
        from langchain.schema.output_parser import StrOutputParser

        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.3,
            api_key=api_key,
        )

        audience_instructions = {
            "autoridades": "Escreva um boletim técnico formal para autoridades governamentais e gestores de emergência. Use linguagem técnica mas clara. Inclua recomendações de ação específicas.",
            "populacao": "Escreva um alerta em linguagem simples e acessível para a população geral. Evite jargões técnicos. Seja direto sobre o que as pessoas devem fazer.",
            "tecnico": "Escreva uma análise técnica detalhada para cientistas e engenheiros de dados. Inclua métricas, comparações históricas e metodologia de detecção.",
        }

        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""Você é o OrbitClima, um sistema de IA especializado em monitoramento climático
            baseado em dados de satélites da NASA. Seu papel é transformar dados técnicos em informações
            úteis e acionáveis.

            {audience_instructions.get(audience, audience_instructions['autoridades'])}

            Sempre mencione que os dados provêm de satélites via NASA POWER API.
            Estruture o boletim com: Resumo Executivo, Situação Atual, Eventos Notáveis,
            Previsão, e Recomendações."""),
            ("human", "Com base nos seguintes dados climáticos satelitais, gere o boletim:\n\n{context}"),
        ])

        chain = prompt | llm | StrOutputParser()
        context = _format_data_for_prompt(df, location, anomaly_report, forecast_df)
        result = chain.invoke({"context": context})
        return result

    except ImportError:
        return _generate_demo_bulletin(df, location, anomaly_report, audience)
    except Exception as e:
        return f"Erro ao gerar relatório via IA: {e}\n\n{_generate_demo_bulletin(df, location, anomaly_report, audience)}"


def _generate_demo_bulletin(
    df: pd.DataFrame,
    location: str,
    anomaly_report: str,
    audience: str,
) -> str:
    """
    Gera boletim de demonstração sem necessidade de API Key.
    Simula saída típica de um LLM para fins de POC.
    """
    recent = df.tail(30)
    n_anomalies = df["anomalia"].sum() if "anomalia" in df.columns else 0
    risk_dist = df["nivel_risco"].value_counts() if "nivel_risco" in df.columns else {}

    bulletin = f"""
╔══════════════════════════════════════════════════════════════════╗
║          🛰️  ORBITCLIMA — BOLETIM CLIMÁTICO SATELITAL            ║
║                    [MODO DEMONSTRAÇÃO — SEM API KEY]             ║
╚══════════════════════════════════════════════════════════════════╝

📍 LOCALIZAÇÃO: {location}
📅 DATA DO RELATÓRIO: {pd.Timestamp.today().strftime('%d/%m/%Y às %H:%M')}
🌐 FONTE DOS DADOS: NASA POWER API (Dados Satelitais)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 RESUMO EXECUTIVO
──────────────────
O sistema OrbitClima processou {len(df)} registros diários de dados satelitais
para {location}. A análise revelou {n_anomalies} eventos anômalos no período,
representando {n_anomalies/len(df)*100:.1f}% dos dias monitorados.

🌡️ SITUAÇÃO ATUAL (últimos 30 dias)
─────────────────────────────────
• Temperatura média: {recent['T2M'].mean():.1f}°C
• Precipitação acumulada: {recent['PRECTOTCORR'].sum():.1f} mm
• Umidade relativa: {recent['RH2M'].mean():.1f}%
• Velocidade do vento: {recent['WS10M'].mean():.1f} m/s

🚨 EVENTOS NOTÁVEIS
──────────────────
{anomaly_report}

📈 DISTRIBUIÇÃO DE RISCO (período completo)
──────────────────────────────────────────
{chr(10).join([f"  • {k}: {v} dias" for k, v in risk_dist.items()]) if risk_dist else '  Classificação não disponível'}

🔮 TENDÊNCIAS E PREVISÃO
────────────────────────
• Temperatura máxima histórica: {df['T2M_MAX'].max():.1f}°C
• Precipitação máxima em 24h: {df['PRECTOTCORR'].max():.1f} mm
• Tendência de temperatura (30d): {"↑ Aquecimento" if recent['T2M'].mean() > df['T2M'].mean() else "↓ Resfriamento"}

⚠️ RECOMENDAÇÕES
────────────────
Com base nos dados satelitais analisados pelo OrbitClima:

1. DEFESA CIVIL: Manter equipes em alerta para os dias com risco ALTO e CRÍTICO identificados.
2. SAÚDE PÚBLICA: Reforçar campanhas de hidratação e proteção solar em períodos de temperatura extrema.
3. AGRICULTURA: Ajustar calendário de irrigação com base nas previsões de precipitação.
4. INFRAESTRUTURA: Verificar sistemas de drenagem antes dos períodos de alta precipitação previstos.
5. POPULAÇÃO: Acompanhar alertas do OrbitClima e das autoridades locais.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🛰️ Este boletim foi gerado automaticamente pelo sistema OrbitClima
   utilizando dados da NASA POWER API (satélites MERRA-2 e CERES).
   Em produção, este texto é gerado por GPT-4o via LangChain.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    return bulletin


def generate_alert_sms(df: pd.DataFrame, location: str) -> str:
    """
    Gera alerta curto em formato SMS/notificação push para a população.
    """
    if "nivel_risco" not in df.columns:
        return "Dados insuficientes para gerar alerta."

    critical_days = df[df["nivel_risco"] == "CRÍTICO"]
    high_days = df[df["nivel_risco"] == "ALTO RISCO"]

    if not critical_days.empty:
        return (
            f"⚠️ ALERTA CRÍTICO OrbitClima — {location}\n"
            f"Evento climático extremo detectado via satélite.\n"
            f"Temperatura: {critical_days['T2M'].max():.1f}°C | "
            f"Chuva: {critical_days['PRECTOTCORR'].max():.0f}mm\n"
            f"Siga orientações da Defesa Civil. #OrbitClima"
        )
    elif not high_days.empty:
        return (
            f"⚠️ ATENÇÃO OrbitClima — {location}\n"
            f"Condições climáticas adversas detectadas.\n"
            f"Fique atento e evite áreas de risco. #OrbitClima"
        )
    else:
        return f"✅ OrbitClima — {location}: Condições climáticas dentro da normalidade. #OrbitClima"


if __name__ == "__main__":
    import numpy as np

    # Dados simulados para demonstração
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=365)
    df_demo = pd.DataFrame({
        "T2M": np.random.normal(28, 6, 365),
        "T2M_MAX": np.random.normal(33, 5, 365),
        "T2M_MIN": np.random.normal(22, 5, 365),
        "PRECTOTCORR": np.abs(np.random.exponential(8, 365)),
        "RH2M": np.random.normal(70, 12, 365),
        "WS10M": np.abs(np.random.normal(4, 2, 365)),
        "anomalia": np.random.choice([True, False], 365, p=[0.05, 0.95]),
        "nivel_risco": np.random.choice(
            ["NORMAL", "ATENÇÃO", "ALTO RISCO", "CRÍTICO"],
            365,
            p=[0.70, 0.20, 0.08, 0.02],
        ),
    }, index=dates)

    anomaly_text = "3 ondas de calor detectadas (Jan, Mar, Nov). 1 episódio de chuva extrema (Fev: 145mm/dia)."

    bulletin = generate_climate_bulletin(
        df=df_demo,
        location="São Paulo, SP",
        anomaly_report=anomaly_text,
        audience="autoridades",
    )
    print(bulletin)

    sms = generate_alert_sms(df_demo.tail(7), "São Paulo, SP")
    print("\n📱 ALERTA SMS/PUSH:")
    print(sms)
