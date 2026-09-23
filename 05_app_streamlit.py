"""
05_app_streamlit.py

App interactiva de escenarios de planeacion de personal (Workforce Management).

Toma el pronostico de volumen de contactos (90 dias, generado por
03_modelo_forecasting.py) y permite simular distintos supuestos operativos
-crecimiento adicional de volumen, AHT, horas de operacion, nivel de servicio
objetivo y shrinkage- para ver en tiempo real el impacto en la dotacion (FTE)
requerida. Es el mismo tipo de analisis "what-if" que se usa en planeacion de
personal real para preparar distintos escenarios de negocio.

Ejecutar con:
    streamlit run 05_app_streamlit.py
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path
from wfm_utils import dotacion_diaria

BASE = Path(__file__).parent
DATA = BASE / "data"

st.set_page_config(page_title="Escenarios de Dotación WFM", layout="wide")

COLOR_PRIMARY = "#1F3864"
COLOR_ACCENT = "#2E75B6"
COLOR_WARN = "#B4720F"

# ---------- Carga de datos ----------
@st.cache_data
def cargar_datos():
    forecast = pd.read_csv(DATA / "forecast_90_dias.csv", parse_dates=["fecha"])
    metricas = pd.read_csv(DATA / "metricas_validacion.csv")
    return forecast, metricas

forecast, metricas = cargar_datos()
mape = metricas.loc[metricas["metrica"] == "MAPE", "valor"].values[0]

# ---------- Encabezado ----------
st.title("📊 Escenarios de Pronóstico y Dotación de Personal")
st.caption(
    "Proyecto de portafolio — datos de volumen de contactos **simulados** (no reales de "
    "ninguna empresa). El modelo de pronóstico (SARIMAX) fue validado con un MAPE de "
    f"**{mape:.1f}%** sobre 60 días fuera de muestra. El cálculo de dotación usa el modelo "
    "Erlang C, estándar de la industria de centros de contacto."
)

# ---------- Panel de supuestos (sidebar) ----------
st.sidebar.header("⚙️ Supuestos del escenario")

crecimiento_extra = st.sidebar.slider(
    "Crecimiento adicional de volumen (%)", -30, 100, 0, step=5,
    help="Simula una campaña, temporada alta, o una caída de demanda no capturada por el modelo histórico."
)
aht_seg = st.sidebar.slider("AHT — Tiempo promedio de atención (segundos)", 120, 900, 360, step=15)
horas_operacion = st.sidebar.slider("Horas de operación por día", 6, 24, 10)
ns_objetivo = st.sidebar.slider("Nivel de Servicio objetivo (%)", 50, 95, 80, step=5) / 100
tiempo_objetivo_seg = st.sidebar.slider("Tiempo objetivo de respuesta (segundos)", 10, 60, 20, step=5)
shrinkage = st.sidebar.slider("Shrinkage (%)", 10, 50, 32, step=1) / 100

st.sidebar.markdown("---")
st.sidebar.caption(
    "**Shrinkage**: proporción del tiempo pagado que un agente NO está disponible para "
    "atender contactos (ausentismo, formación, pausas, tareas administrativas)."
)

# ---------- Cálculo del escenario ----------
df = forecast.copy()
df["volumen_escenario"] = df["volumen_pronosticado"] * (1 + crecimiento_extra / 100)

resultados = df["volumen_escenario"].apply(
    lambda v: dotacion_diaria(v, horas_operacion, aht_seg, ns_objetivo, tiempo_objetivo_seg, shrinkage)
)
df["contactos_hora"] = resultados.apply(lambda r: r["contactos_hora"])
df["agentes_base"] = resultados.apply(lambda r: r["agentes_base"])
df["fte_requerido"] = resultados.apply(lambda r: r["fte_con_shrinkage"])

# Escenario base (0% crecimiento, supuestos por defecto) para comparar
resultados_base = forecast["volumen_pronosticado"].apply(
    lambda v: dotacion_diaria(v, 10, 360, 0.80, 20, 0.32)
)
df["fte_base"] = resultados_base.apply(lambda r: r["fte_con_shrinkage"])

# ---------- KPIs ----------
col1, col2, col3, col4 = st.columns(4)
col1.metric("FTE promedio (escenario)", f"{df['fte_requerido'].mean():.1f}",
            delta=f"{df['fte_requerido'].mean() - df['fte_base'].mean():+.1f} vs. caso base")
col2.metric("FTE pico", f"{df['fte_requerido'].max()}")
col3.metric("FTE mínimo", f"{df['fte_requerido'].min()}")
col4.metric("Volumen promedio/día", f"{df['volumen_escenario'].mean():,.0f}".replace(",", "."))

st.markdown("---")

# ---------- Gráfico: volumen y dotación ----------
fig = go.Figure()
fig.add_trace(go.Bar(
    x=df["fecha"], y=df["fte_requerido"], name="FTE requerido (escenario)",
    marker_color=COLOR_ACCENT, opacity=0.85, yaxis="y1"
))
fig.add_trace(go.Scatter(
    x=df["fecha"], y=df["fte_base"], name="FTE caso base",
    line=dict(color=COLOR_WARN, dash="dot", width=2), yaxis="y1"
))
fig.add_trace(go.Scatter(
    x=df["fecha"], y=df["volumen_escenario"], name="Volumen pronosticado",
    line=dict(color=COLOR_PRIMARY, width=2), yaxis="y2"
))
fig.update_layout(
    title="Volumen pronosticado y dotación (FTE) requerida — 90 días",
    yaxis=dict(title="FTE requerido"),
    yaxis2=dict(title="Volumen/día", overlaying="y", side="right"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    height=460,
    margin=dict(t=60),
)
st.plotly_chart(fig, use_container_width=True)

# ---------- Tabla de detalle ----------
with st.expander("📋 Ver detalle diario del escenario"):
    tabla = df[["fecha", "volumen_escenario", "contactos_hora", "agentes_base", "fte_requerido"]].copy()
    tabla.columns = ["Fecha", "Volumen", "Contactos/hora", "Agentes base (Erlang C)", "FTE (con shrinkage)"]
    tabla["Volumen"] = tabla["Volumen"].round(0).astype(int)
    st.dataframe(tabla, use_container_width=True, hide_index=True)

st.markdown("---")
st.caption(
    "Metodología: SARIMAX(1,0,1)(1,1,1)[7] con términos de Fourier para estacionalidad anual "
    "→ pronóstico de volumen → modelo Erlang C para dotación por Nivel de Servicio objetivo → "
    "ajuste por shrinkage. Código completo y notebook en el repositorio de GitHub."
)
