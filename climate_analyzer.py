"""
climate_analyzer.py
--------------------
Módulo de análise climática com Machine Learning.

Implementa dois modelos principais:
1. Isolation Forest — detecção de anomalias climáticas (eventos extremos)
2. LSTM (Long Short-Term Memory) — previsão de temperatura a curto prazo

Disciplinas contempladas: Machine Learning, Deep Learning, Análise de Dados
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
import warnings

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
#  MODELO 1: Detecção de Anomalias Climáticas
# ─────────────────────────────────────────────

class ClimateAnomalyDetector:
    """
    Utiliza Isolation Forest para identificar anomalias climáticas —
    eventos fora do padrão histórico, como ondas de calor, secas extremas
    e chuvas torrenciais detectadas via dados de satélite.
    """

    def __init__(self, contamination: float = 0.05):
        """
        Args:
            contamination: Proporção esperada de anomalias no dataset (5% padrão).
        """
        self.model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=200,
        )
        self.scaler = MinMaxScaler()
        self.feature_cols = []
        self.is_trained = False

    def fit(self, df: pd.DataFrame, feature_cols: list = None) -> "ClimateAnomalyDetector":
        """
        Treina o modelo com dados históricos.

        Args:
            df: DataFrame com dados climáticos
            feature_cols: Colunas a usar como features (padrão: todas numéricas)
        """
        self.feature_cols = feature_cols or df.select_dtypes(include=[np.number]).columns.tolist()
        X = df[self.feature_cols].dropna()
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        self.is_trained = True
        print(f"[AnomalyDetector] ✅ Treinado com {len(X)} amostras e {len(self.feature_cols)} features.")
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detecta anomalias no DataFrame.

        Retorna:
            DataFrame com coluna 'anomalia' (True/False) e 'score_anomalia'.
        """
        if not self.is_trained:
            raise RuntimeError("Modelo não treinado. Execute .fit() primeiro.")

        X = df[self.feature_cols].fillna(df[self.feature_cols].mean())
        X_scaled = self.scaler.transform(X)

        predictions = self.model.predict(X_scaled)
        scores = self.model.score_samples(X_scaled)

        result = df.copy()
        result["anomalia"] = predictions == -1  # -1 = anomalia no Isolation Forest
        result["score_anomalia"] = -scores       # Inverter: maior score = mais anômalo

        n_anomalias = result["anomalia"].sum()
        print(f"[AnomalyDetector] 🚨 {n_anomalias} anomalias detectadas em {len(result)} registros ({n_anomalias/len(result)*100:.1f}%)")

        return result

    def get_anomaly_report(self, df_with_anomalies: pd.DataFrame) -> str:
        """
        Gera resumo textual das anomalias detectadas.
        """
        anomalias = df_with_anomalies[df_with_anomalies["anomalia"]]
        if anomalias.empty:
            return "Nenhuma anomalia climática detectada no período analisado."

        report_lines = [
            f"Total de anomalias: {len(anomalias)} eventos",
            f"Período com mais anomalias: {anomalias.index.to_period('M').value_counts().idxmax()}",
        ]

        if "T2M" in anomalias.columns:
            max_temp = anomalias["T2M"].max()
            min_temp = anomalias["T2M"].min()
            report_lines.append(f"Temperatura extrema máxima registrada: {max_temp:.1f}°C")
            report_lines.append(f"Temperatura extrema mínima registrada: {min_temp:.1f}°C")

        if "PRECTOTCORR" in anomalias.columns:
            max_prec = anomalias["PRECTOTCORR"].max()
            report_lines.append(f"Precipitação máxima em evento anômalo: {max_prec:.1f} mm/dia")

        return "\n".join(report_lines)


# ─────────────────────────────────────────────
#  MODELO 2: Previsão de Temperatura com LSTM
# ─────────────────────────────────────────────

class TemperatureForecaster:
    """
    Rede LSTM para previsão de temperatura a curto prazo (7 a 30 dias),
    utilizando séries temporais de dados satelitais.
    """

    def __init__(self, lookback: int = 30, forecast_days: int = 7):
        """
        Args:
            lookback: Janela de dias históricos usados como input (padrão: 30 dias)
            forecast_days: Quantos dias à frente prever (padrão: 7 dias)
        """
        self.lookback = lookback
        self.forecast_days = forecast_days
        self.scaler = MinMaxScaler()
        self.model = None
        self.history = None

    def _build_model(self, input_shape):
        """Constrói a arquitetura LSTM."""
        try:
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
            from tensorflow.keras.callbacks import EarlyStopping
        except ImportError:
            raise ImportError("TensorFlow necessário. Execute: pip install tensorflow")

        model = Sequential([
            Bidirectional(LSTM(64, return_sequences=True), input_shape=input_shape),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(self.forecast_days),
        ])
        model.compile(optimizer="adam", loss="huber", metrics=["mae"])
        return model

    def _create_sequences(self, data: np.ndarray):
        """Cria sequências de entrada e saída para o LSTM."""
        X, y = [], []
        for i in range(len(data) - self.lookback - self.forecast_days + 1):
            X.append(data[i : i + self.lookback])
            y.append(data[i + self.lookback : i + self.lookback + self.forecast_days, 0])
        return np.array(X), np.array(y)

    def fit(self, df: pd.DataFrame, target_col: str = "T2M", epochs: int = 50) -> "TemperatureForecaster":
        """
        Treina o modelo LSTM com dados de temperatura.

        Args:
            df: DataFrame com dados climáticos
            target_col: Coluna alvo para previsão (padrão: temperatura a 2m)
            epochs: Número de épocas de treino
        """
        from tensorflow.keras.callbacks import EarlyStopping

        features = [c for c in ["T2M", "T2M_MAX", "T2M_MIN", "RH2M", "PRECTOTCORR"] if c in df.columns]
        data = df[features].dropna().values
        data_scaled = self.scaler.fit_transform(data)

        X, y = self._create_sequences(data_scaled)
        split = int(len(X) * 0.8)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        self.model = self._build_model((self.lookback, X.shape[2]))

        early_stop = EarlyStopping(patience=10, restore_best_weights=True)
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            callbacks=[early_stop],
            verbose=1,
        )
        print(f"[LSTM] ✅ Treinamento concluído. MAE final: {min(self.history.history['val_mae']):.3f}°C")
        return self

    def forecast(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Gera previsão para os próximos N dias.

        Returns:
            DataFrame com datas futuras e temperaturas previstas.
        """
        if self.model is None:
            raise RuntimeError("Modelo não treinado. Execute .fit() primeiro.")

        features = [c for c in ["T2M", "T2M_MAX", "T2M_MIN", "RH2M", "PRECTOTCORR"] if c in df.columns]
        data = df[features].dropna().values
        data_scaled = self.scaler.transform(data)

        last_sequence = data_scaled[-self.lookback:].reshape(1, self.lookback, -1)
        pred_scaled = self.model.predict(last_sequence, verbose=0)[0]

        # Inverter normalização apenas para a feature de temperatura (índice 0)
        dummy = np.zeros((self.forecast_days, data_scaled.shape[1]))
        dummy[:, 0] = pred_scaled
        pred_original = self.scaler.inverse_transform(dummy)[:, 0]

        last_date = df.dropna().index[-1]
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=self.forecast_days,
            freq="D",
        )

        forecast_df = pd.DataFrame({
            "data": future_dates,
            "temperatura_prevista_C": pred_original,
        }).set_index("data")

        return forecast_df


# ─────────────────────────────────────────────
#  CLASSIFICAÇÃO DE RISCO CLIMÁTICO
# ─────────────────────────────────────────────

def classify_climate_risk(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classifica o risco climático de cada dia com base em regras heurísticas
    derivadas de limiares científicos reconhecidos.

    Níveis:
        NORMAL — condições dentro do padrão histórico
        ATENÇÃO — valores moderadamente extremos
        ALTO RISCO — condições com potencial de impacto
        CRÍTICO — eventos extremos: enchentes, ondas de calor, seca severa
    """
    df = df.copy()
    df["nivel_risco"] = "NORMAL"

    # Regras baseadas em limiares científicos (OMM / INMET)
    if "T2M" in df.columns:
        df.loc[df["T2M"] > 35, "nivel_risco"] = "ATENÇÃO"
        df.loc[df["T2M"] > 40, "nivel_risco"] = "ALTO RISCO"
        df.loc[df["T2M"] > 44, "nivel_risco"] = "CRÍTICO"
        df.loc[df["T2M"] < 0, "nivel_risco"] = "ATENÇÃO"

    if "PRECTOTCORR" in df.columns:
        df.loc[df["PRECTOTCORR"] > 50, "nivel_risco"] = "ATENÇÃO"
        df.loc[df["PRECTOTCORR"] > 100, "nivel_risco"] = "ALTO RISCO"
        df.loc[df["PRECTOTCORR"] > 150, "nivel_risco"] = "CRÍTICO"

    if "WS10M" in df.columns:
        df.loc[df["WS10M"] > 15, "nivel_risco"] = "ATENÇÃO"
        df.loc[df["WS10M"] > 25, "nivel_risco"] = "CRÍTICO"

    return df


if __name__ == "__main__":
    # Exemplo de uso com dados simulados (sem necessidade de API)
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=365)
    df_mock = pd.DataFrame({
        "T2M": np.random.normal(28, 5, 365),
        "PRECTOTCORR": np.abs(np.random.normal(5, 10, 365)),
        "RH2M": np.random.normal(70, 15, 365),
        "WS10M": np.abs(np.random.normal(3, 2, 365)),
    }, index=dates)

    # Injetar algumas anomalias artificiais
    df_mock.iloc[50, 0] = 48.5   # Onda de calor
    df_mock.iloc[120, 1] = 180.0  # Enchente

    # Detectar anomalias
    detector = ClimateAnomalyDetector(contamination=0.05)
    detector.fit(df_mock)
    result = detector.predict(df_mock)

    print("\nAnomalias detectadas:")
    print(result[result["anomalia"]][["T2M", "PRECTOTCORR", "score_anomalia"]])

    # Classificar risco
    result = classify_climate_risk(result)
    print("\nDistribuição de risco:")
    print(result["nivel_risco"].value_counts())
