"""
04_calculo_dotacion.py

Traduce el pronostico de volumen (90 dias) en dotacion diaria requerida (FTE),
usando el modelo Erlang C de wfm_utils.py con supuestos base (caso "actual"):

    AHT (tiempo promedio de atencion): 360 segundos (6 minutos)
    Horas de operacion:                10 horas/dia
    Nivel de servicio objetivo:        80% de contactos atendidos
    Tiempo objetivo de respuesta:      20 segundos
    Shrinkage:                         32% (ausentismo, pausas, formacion, adm.)

Estos supuestos son ajustables interactivamente en la app de Streamlit
(05_app_streamlit.py) para simular escenarios.
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from wfm_utils import dotacion_diaria

BASE = Path(__file__).parent
DATA = BASE / "data"
OUT = BASE / "outputs"

# --- Supuestos del caso base ---
AHT_SEG = 360
HORAS_OPERACION = 10
NS_OBJETIVO = 0.80
TIEMPO_OBJETIVO_SEG = 20
SHRINKAGE = 0.32

df = pd.read_csv(DATA / "forecast_90_dias.csv", parse_dates=["fecha"])

resultados = df["volumen_pronosticado"].apply(
    lambda v: dotacion_diaria(v, HORAS_OPERACION, AHT_SEG, NS_OBJETIVO, TIEMPO_OBJETIVO_SEG, SHRINKAGE)
)
df["contactos_hora"] = resultados.apply(lambda r: r["contactos_hora"])
df["agentes_base"] = resultados.apply(lambda r: r["agentes_base"])
df["fte_requerido"] = resultados.apply(lambda r: r["fte_con_shrinkage"])

df.to_csv(DATA / "dotacion_90_dias.csv", index=False)

print("=== Resumen de dotacion proyectada (proximos 90 dias) ===")
print(f"FTE promedio requerido: {df['fte_requerido'].mean():.1f}")
print(f"FTE minimo (dia mas bajo): {df['fte_requerido'].min()}")
print(f"FTE maximo (dia mas alto): {df['fte_requerido'].max()}")
print(f"\nSupuestos: AHT={AHT_SEG}s | Horas op.={HORAS_OPERACION}h | "
      f"NS objetivo={NS_OBJETIVO:.0%} en {TIEMPO_OBJETIVO_SEG}s | Shrinkage={SHRINKAGE:.0%}")

# --- Grafico de dotacion proyectada ---
fig, ax1 = plt.subplots(figsize=(12, 5))
ax1.bar(df["fecha"], df["fte_requerido"], color="#2E75B6", alpha=0.75, label="FTE requerido")
ax1.set_ylabel("FTE requerido", color="#1F3864")
ax1.set_title("Dotacion proyectada (FTE) - proximos 90 dias")
ax2 = ax1.twinx()
ax2.plot(df["fecha"], df["volumen_pronosticado"], color="#B4720F", linewidth=1.5, label="Volumen pronosticado")
ax2.set_ylabel("Volumen pronosticado (contactos/dia)", color="#B4720F")
fig.tight_layout()
plt.savefig(OUT / "07_dotacion_proyectada.png", dpi=130)
plt.close()

print(f"\nGuardado en: {DATA / 'dotacion_90_dias.csv'}")
print(f"Figura: {OUT / '07_dotacion_proyectada.png'}")
