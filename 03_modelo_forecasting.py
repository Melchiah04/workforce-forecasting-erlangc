"""
03_modelo_forecasting.py

Modelo de pronostico de volumen de contactos con SARIMAX (statsmodels):
- Estacionalidad semanal modelada con el componente estacional SARIMA (periodo=7)
- Estacionalidad anual modelada con terminos de Fourier (seno/coseno del dia del
  anio) como variables exogenas, tecnica estandar cuando una serie tiene mas de
  un ciclo estacional y el periodo anual (365) es demasiado largo para el
  componente estacional nativo de SARIMA.
- Validacion con holdout de los ultimos 60 dias (fuera de muestra).
- Metricas: MAPE y RMSE.
- Pronostico extendido 90 dias hacia adelante para alimentar la app de escenarios.

NOTA TECNICA (diagnostico real durante el desarrollo):
Una primera version con doble diferenciacion (d=1 regular + D=1 estacional)
producia un MAPE de ~174% en el holdout: el pronostico "explotaba" exponencialmente
en horizontes largos (60 dias), un efecto clasico de sobre-diferenciar una serie
que ya tiene una tendencia predominantemente lineal. La solucion fue modelar la
tendencia de forma explicita con el parametro trend="t" de SARIMAX, dejando solo
la diferenciacion estacional (D=1, periodo=7) para la estacionalidad semanal
(d=0 en la parte regular). Este cambio bajo el MAPE de validacion a ~12%.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.tsa.statespace.sarimax import SARIMAX

BASE = Path(__file__).parent
OUT = BASE / "outputs"
DATA = BASE / "data"
OUT.mkdir(exist_ok=True)

df = pd.read_csv(DATA / "volumen_contactos.csv", parse_dates=["fecha"])
df = df.set_index("fecha").asfreq("D")
serie = df["volumen_contactos"]


def fourier_terms(index, periodo=365.25, n_armonicos=2):
    """Terminos de Fourier (seno/coseno) para modelar estacionalidad anual como exogena."""
    dia_del_anio = index.dayofyear.values
    terms = {}
    for k in range(1, n_armonicos + 1):
        terms[f"sin_{k}"] = np.sin(2 * np.pi * k * dia_del_anio / periodo)
        terms[f"cos_{k}"] = np.cos(2 * np.pi * k * dia_del_anio / periodo)
    return pd.DataFrame(terms, index=index)


exog_completo = fourier_terms(serie.index)

# --- Split train / test (holdout de 60 dias) ---
HOLDOUT = 60
train_y, test_y = serie.iloc[:-HOLDOUT], serie.iloc[-HOLDOUT:]
train_x, test_x = exog_completo.iloc[:-HOLDOUT], exog_completo.iloc[-HOLDOUT:]

modelo = SARIMAX(
    train_y,
    exog=train_x,
    order=(1, 0, 1),
    seasonal_order=(1, 1, 1, 7),
    trend="t",
    enforce_stationarity=False,
    enforce_invertibility=False,
)
resultado = modelo.fit(disp=False, maxiter=200)
print(resultado.summary().tables[0])

# --- Validacion en holdout ---
pred = resultado.get_forecast(steps=HOLDOUT, exog=test_x)
pred_media = pred.predicted_mean
pred_ic = pred.conf_int(alpha=0.10)  # intervalo de confianza 90%

mape = float(np.mean(np.abs((test_y - pred_media) / test_y)) * 100)
rmse = float(np.sqrt(np.mean((test_y - pred_media) ** 2)))
print(f"\n=== Validacion (holdout {HOLDOUT} dias) ===")
print(f"MAPE: {mape:.2f}%")
print(f"RMSE: {rmse:.1f} contactos/dia")

# --- Grafico de validacion ---
fig, ax = plt.subplots(figsize=(12, 5))
train_y.iloc[-120:].plot(ax=ax, label="Historico (train)", color="#5B6577")
test_y.plot(ax=ax, label="Real (holdout)", color="#1F3864", linewidth=2)
pred_media.plot(ax=ax, label="Pronostico", color="#2E75B6", linewidth=2, linestyle="--")
ax.fill_between(pred_ic.index, pred_ic.iloc[:, 0], pred_ic.iloc[:, 1],
                 color="#2E75B6", alpha=0.15, label="Intervalo de confianza 90%")
ax.set_title(f"Validacion del modelo (MAPE={mape:.1f}%, RMSE={rmse:.0f})")
ax.set_ylabel("Contactos/dia")
ax.legend()
plt.tight_layout()
plt.savefig(OUT / "05_validacion_forecast.png", dpi=130)
plt.close()

# --- Reentrenar con TODA la serie y proyectar 90 dias hacia adelante ---
modelo_full = SARIMAX(
    serie,
    exog=exog_completo,
    order=(1, 0, 1),
    seasonal_order=(1, 1, 1, 7),
    trend="t",
    enforce_stationarity=False,
    enforce_invertibility=False,
)
resultado_full = modelo_full.fit(disp=False, maxiter=200)

FUTURO = 90
fechas_futuras = pd.date_range(serie.index[-1] + pd.Timedelta(days=1), periods=FUTURO, freq="D")
exog_futuro = fourier_terms(fechas_futuras)
pred_futuro = resultado_full.get_forecast(steps=FUTURO, exog=exog_futuro)
media_futuro = pred_futuro.predicted_mean
ic_futuro = pred_futuro.conf_int(alpha=0.10)

df_forecast = pd.DataFrame({
    "fecha": fechas_futuras,
    "volumen_pronosticado": media_futuro.values,
    "ic_inferior_90": ic_futuro.iloc[:, 0].values,
    "ic_superior_90": ic_futuro.iloc[:, 1].values,
})
df_forecast["volumen_pronosticado"] = df_forecast["volumen_pronosticado"].clip(lower=0)
df_forecast["ic_inferior_90"] = df_forecast["ic_inferior_90"].clip(lower=0)
df_forecast.to_csv(DATA / "forecast_90_dias.csv", index=False)

# Guardar tambien metricas de validacion para mostrarlas en la app / README
pd.DataFrame([{"metrica": "MAPE", "valor": round(mape, 2)},
              {"metrica": "RMSE", "valor": round(rmse, 1)},
              {"metrica": "dias_holdout", "valor": HOLDOUT}]).to_csv(
    DATA / "metricas_validacion.csv", index=False
)

# --- Grafico del pronostico futuro ---
fig, ax = plt.subplots(figsize=(12, 5))
serie.iloc[-150:].plot(ax=ax, label="Historico", color="#5B6577")
media_futuro.plot(ax=ax, label="Pronostico (90 dias)", color="#1F7A4D", linewidth=2)
ax.fill_between(fechas_futuras, ic_futuro.iloc[:, 0], ic_futuro.iloc[:, 1],
                 color="#1F7A4D", alpha=0.15, label="Intervalo de confianza 90%")
ax.set_title("Pronostico de volumen de contactos - proximos 90 dias")
ax.set_ylabel("Contactos/dia")
ax.legend()
plt.tight_layout()
plt.savefig(OUT / "06_pronostico_90_dias.png", dpi=130)
plt.close()

print(f"\nPronostico guardado en: {DATA / 'forecast_90_dias.csv'}")
print(f"Figuras guardadas en: {OUT}")
