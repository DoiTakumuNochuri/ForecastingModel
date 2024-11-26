import pandas as pd
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def forecast_sarimax(data, freq, future_periods):
    # 必須列の確認
    required_columns = ['Date', 'Value']
    if not all(col in data.columns for col in required_columns):
        raise ValueError(f"Excel ファイルに以下の列名が必要です: {required_columns}")
    
    # データ前処理
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
    data.dropna(subset=['Date'], inplace=True)
    data.set_index('Date', inplace=True)
    ts = data['Value']
    
    # リサンプリング
    ts = ts.resample(freq).mean().fillna(method='ffill')
    if ts.empty:
        raise ValueError("リサンプリング後のデータが空です。")
    
    # モデルの最適化
    model = auto_arima(ts, seasonal=True, m=12 if freq in ['M', 'ME'] else 1, trace=True, error_action='ignore', suppress_warnings=True, stepwise=True)
    
    # SARIMAXのフィッティング
    sarimax_model = SARIMAX(ts, order=model.order, seasonal_order=model.seasonal_order)
    results = sarimax_model.fit(disp=False)
    
    # 未来予測
    forecast = results.get_forecast(steps=future_periods)
    forecast_values = forecast.predicted_mean
    forecast_ci = forecast.conf_int()

    # グラフのプロット
    plt.figure(figsize=(12, 6))
    plt.plot(ts, label="Observed", color='blue')
    forecast_index = pd.date_range(start=ts.index[-1] + pd.offsets.DateOffset(1), periods=future_periods, freq=freq)
    plt.plot(forecast_index, forecast_values, label="Forecast", color='red')
    plt.fill_between(forecast_index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], color='pink', alpha=0.3)
    plt.legend()
    plt.title("SARIMAX Forecast")
    plt.show()
    
    # 結果を返却
    forecast_df = pd.DataFrame({
        "Date": forecast_index,
        "Forecast": forecast_values,
        "Lower CI": forecast_ci.iloc[:, 0],
        "Upper CI": forecast_ci.iloc[:, 1]
    })
    return forecast_df
