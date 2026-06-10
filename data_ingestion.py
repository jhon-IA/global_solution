"""
data_ingestion.py
-----------------
Módulo responsável por coletar dados climáticos da NASA POWER API.

A NASA POWER (Prediction of Worldwide Energy Resources) disponibiliza dados
derivados de satélites para temperatura, precipitação, irradiação solar,
velocidade do vento, umidade relativa e muito mais — gratuitamente e sem autenticação.

Documentação: https://power.larc.nasa.gov/api/temporal/
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional


# Parâmetros disponíveis na NASA POWER API
NASA_PARAMETERS = {
    "T2M": "Temperatura a 2m (°C)",
    "PRECTOTCORR": "Precipitação (mm/dia)",
    "WS10M": "Velocidade do Vento a 10m (m/s)",
    "RH2M": "Umidade Relativa a 2m (%)",
    "ALLSKY_SFC_SW_DWN": "Irradiação Solar Global (kWh/m²/dia)",
    "T2M_MAX": "Temperatura Máxima a 2m (°C)",
    "T2M_MIN": "Temperatura Mínima a 2m (°C)",
}

BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"


def fetch_climate_data(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    parameters: Optional[list] = None,
) -> pd.DataFrame:
    """
    Coleta dados climáticos históricos de uma localização específica via NASA POWER API.

    Args:
        latitude: Latitude do ponto de interesse (-90 a 90)
        longitude: Longitude do ponto de interesse (-180 a 180)
        start_date: Data inicial no formato 'YYYYMMDD'
        end_date: Data final no formato 'YYYYMMDD'
        parameters: Lista de parâmetros a coletar (padrão: todos os principais)

    Returns:
        DataFrame pandas com os dados climáticos indexados por data.
    """
    if parameters is None:
        parameters = list(NASA_PARAMETERS.keys())

    params = {
        "parameters": ",".join(parameters),
        "community": "RE",
        "longitude": longitude,
        "latitude": latitude,
        "start": start_date,
        "end": end_date,
        "format": "JSON",
    }

    print(f"[OrbitClima] Coletando dados satelitais para lat={latitude}, lon={longitude}")
    print(f"[OrbitClima] Período: {start_date} → {end_date}")

    try:
        response = requests.get(BASE_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        raise ConnectionError(f"Erro ao acessar NASA POWER API: {e}")

    # Extrair os dados dos parâmetros
    records = data["properties"]["parameter"]

    # Converter para DataFrame
    df = pd.DataFrame(records)
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    df.index.name = "data"

    # Substituir valores inválidos (-999) por NaN
    df.replace(-999, float("nan"), inplace=True)

    print(f"[OrbitClima] ✅ {len(df)} registros coletados com sucesso.")
    return df


def fetch_recent_data(
    latitude: float,
    longitude: float,
    days_back: int = 365,
) -> pd.DataFrame:
    """
    Atalho para coletar dados dos últimos N dias.

    Args:
        latitude: Latitude do local
        longitude: Longitude do local
        days_back: Quantos dias retroativos coletar (padrão: 1 ano)

    Returns:
        DataFrame com dados climáticos recentes.
    """
    end = datetime.today() - timedelta(days=7)  # API tem lag de ~7 dias
    start = end - timedelta(days=days_back)

    return fetch_climate_data(
        latitude=latitude,
        longitude=longitude,
        start_date=start.strftime("%Y%m%d"),
        end_date=end.strftime("%Y%m%d"),
    )


def get_brazil_regions() -> dict:
    """
    Retorna coordenadas de regiões-chave do Brasil para monitoramento.
    """
    return {
        "Amazônia (Manaus)": {"lat": -3.10, "lon": -60.02},
        "Pantanal (Corumbá)": {"lat": -19.01, "lon": -57.65},
        "Nordeste Semiárido (Petrolina)": {"lat": -9.38, "lon": -40.51},
        "Sul (Porto Alegre)": {"lat": -30.03, "lon": -51.23},
        "Cerrado (Brasília)": {"lat": -15.78, "lon": -47.93},
        "São Paulo (Metrópole)": {"lat": -23.55, "lon": -46.63},
    }


if __name__ == "__main__":
    # Exemplo: coletar dados de São Paulo dos últimos 2 anos
    df = fetch_climate_data(
        latitude=-23.55,
        longitude=-46.63,
        start_date="20230101",
        end_date="20241231",
    )
    print("\nPrimeiros registros:")
    print(df.head())
    print(f"\nEstatísticas descritivas:")
    print(df.describe())
