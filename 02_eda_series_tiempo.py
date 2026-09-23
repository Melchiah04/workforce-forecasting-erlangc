"""
02_eda_series_tiempo.py

Analisis exploratorio de la serie de tiempo de volumen de contactos:
- Visualizacion general (tendencia + estacionalidad a simple vista)
- Descomposicion estacional (tendencia / estacionalidad / residuo)
- Prueba de estacionariedad (Dickey-Fuller aumentada)
- Autocorrelacion (ACF/PACF) para orientar el orden del modelo SARIMA

Genera figuras en outputs/ para el README y el notebook.
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

BASE = Path(__file__).parent
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)

df = pd.read_csv(BASE / "data" / "volumen_contactos.csv", parse_dates=["fecha"])
df = df.set_index("fecha")
serie = df["volumen_contactos"]

# --- 1. Serie completa ---
fig, ax = plt.subplots(figsize=(12, 4))
serie.plot(ax=ax, color="#1F3864", linewidth=0.8)
ax.set_title("Volumen diario de contactos (2023-2025)")
ax.set_ylabel("Contactos/dia")
plt.tight_layout()
plt.savefig(OUT / "01_serie_completa.png", dpi=130)
plt.close()

# --- 2. Zoom a 3 meses para ver estacionalidad semanal ---
fig, ax = plt.subplots(figsize=(12, 4))
serie.loc["2025-01-01":"2025-03-31"].plot(ax=ax, color="#2E75B6", marker="o", markersize=2)
ax.set_title("Zoom Q1 2025 - Patron semanal visible")
ax.set_ylabel("Contactos/dia")
plt.tight_layout()
plt.savefig(OUT / "02_zoom_trimestre.png", dpi=130)
plt.close()

# --- 3. Descomposicion estacional (periodo semanal = 7) ---
decomposicion = seasonal_decompose(serie, model="multiplicative", period=7)
fig = decomposicion.plot()
fig.set_size_inches(12, 8)
plt.tight_layout()
plt.savefig(OUT / "03_descomposicion.png", dpi=130)
plt.close()

# --- 4. Prueba de estacionariedad (ADF) sobre la serie y su diferencia semanal ---
adf_original = adfuller(serie.dropna())
serie_diff7 = serie.diff(7).dropna()
adf_diff = adfuller(serie_diff7)

print("=== Prueba Dickey-Fuller Aumentada (ADF) ===")
print(f"Serie original     -> estadistico={adf_original[0]:.3f}, p-valor={adf_original[1]:.4f}")
print(f"Diferenciada (d=7)  -> estadistico={adf_diff[0]:.3f}, p-valor={adf_diff[1]:.4f}")
print()
if adf_original[1] > 0.05:
    print("La serie original NO es estacionaria (p > 0.05) -> requiere diferenciacion estacional.")
else:
    print("La serie original es estacionaria (p <= 0.05).")
if adf_diff[1] <= 0.05:
    print("Tras diferenciar con periodo 7 (semanal), la serie SI es estacionaria.")

# --- 5. ACF / PACF sobre la serie diferenciada para orientar el orden SARIMA ---
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(serie_diff7, lags=30, ax=axes[0])
axes[0].set_title("ACF (serie diferenciada, d=7)")
plot_pacf(serie_diff7, lags=30, ax=axes[1], method="ywm")
axes[1].set_title("PACF (serie diferenciada, d=7)")
plt.tight_layout()
plt.savefig(OUT / "04_acf_pacf.png", dpi=130)
plt.close()

print(f"\nFiguras guardadas en: {OUT}")
