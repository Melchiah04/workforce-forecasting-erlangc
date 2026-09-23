# Pronóstico de Volumen y Planeación de Dotación (Workforce Management)

Proyecto end-to-end de forecasting de demanda aplicado a planeación de personal:
desde una serie de tiempo de volumen de contactos hasta la dotación (FTE)
diaria requerida, con una app interactiva para simular escenarios de negocio.

## ⚠️ Nota sobre los datos

Este proyecto usa una serie de tiempo **sintética** (simulada con
`01_generar_datos_sinteticos.py`), construida para comportarse como una serie
real de volumen de contactos de un centro de servicio: tendencia de
crecimiento, estacionalidad semanal (lunes/martes altos, domingo bajo),
estacionalidad anual, feriados colombianos y ruido aleatorio.

**No son datos reales de ninguna empresa** — datos operativos reales nunca
podrían publicarse en un repositorio abierto. La metodología, las técnicas
(SARIMAX, Erlang C) y el criterio aplicado sí son los mismos que usaría sobre
datos reales de operación, con base en mi experiencia gestionando forecasting
y planeación de personal en operaciones de hasta 400 posiciones.

## Contexto

Durante más de 10 años trabajé liderando forecasting y planeación de personal
(Workforce Management) en operaciones de servicio al cliente. Este proyecto
formaliza ese conocimiento operativo con herramientas de ciencia de datos:
un modelo estadístico de series de tiempo para pronosticar demanda, y el
modelo Erlang C (estándar de la industria) para traducir ese pronóstico en
dotación real, en vez de reglas simples de proporcionalidad.

## Metodología

1. **Generación de datos** — serie sintética diaria de 3 años (2023-2025).
2. **Análisis exploratorio** — descomposición estacional, prueba de
   estacionariedad (Dickey-Fuller aumentada), autocorrelación (ACF/PACF).
3. **Modelo de pronóstico** — `SARIMAX(1,0,1)(1,1,1)[7]` con términos de
   Fourier como variables exógenas para la estacionalidad anual. Validado con
   un holdout de 60 días fuera de muestra.
4. **Cálculo de dotación** — modelo Erlang C (`wfm_utils.py`) que, dado un
   volumen, AHT, Nivel de Servicio objetivo y tiempo de respuesta, calcula los
   agentes necesarios; luego se ajusta por shrinkage para llegar al FTE final.
5. **App de escenarios** — dashboard en Streamlit para simular crecimiento de
   volumen, cambios de AHT, NS objetivo y shrinkage, y ver el impacto
   inmediato en la dotación proyectada.

## Resultados

- MAPE de validación (60 días holdout): **~12%**
- Dotación proyectada (90 días, caso base): entre 23 y 39 FTE según el día,
  con un promedio de ~33 FTE.

### Nota técnica honesta (diagnóstico real durante el desarrollo)

La primera versión del modelo, con doble diferenciación (d=1 regular + D=1
estacional), daba un MAPE de ~174%: el pronóstico "explotaba"
exponencialmente en el horizonte de 60 días — un efecto clásico de
sobre-diferenciar una serie con tendencia predominantemente lineal. La
solución fue modelar la tendencia de forma explícita (`trend="t"`) y dejar
solo la diferenciación estacional. Documento este error a propósito: es un
error real de modelado, no un ejemplo de libro de texto, y el proceso de
diagnosticarlo y corregirlo es representativo del trabajo real con series de
tiempo.

## Estructura del repositorio

```
├── 01_generar_datos_sinteticos.py   # Genera la serie de tiempo simulada
├── 02_eda_series_tiempo.py          # Análisis exploratorio
├── 03_modelo_forecasting.py         # Modelo SARIMAX + validación + forecast 90 días
├── wfm_utils.py                     # Funciones de cálculo Erlang C / dotación
├── 04_calculo_dotacion.py           # Aplica Erlang C al forecast (caso base)
├── 05_app_streamlit.py              # App interactiva de escenarios
├── Pronostico_Dotacion_WFM.ipynb    # Notebook end-to-end (ejecutado)
├── data/                            # Datos generados y resultados intermedios
├── outputs/                         # Gráficos generados
└── requirements.txt
```

## Cómo ejecutar

```bash
pip install -r requirements.txt

python 01_generar_datos_sinteticos.py
python 02_eda_series_tiempo.py
python 03_modelo_forecasting.py
python 04_calculo_dotacion.py

# App interactiva de escenarios
streamlit run 05_app_streamlit.py
```

## Stack técnico

Python · pandas · numpy · statsmodels (SARIMAX) · matplotlib · Streamlit ·
Plotly · Jupyter

---

**Autor:** Juan Diego Roncancio Melo — [github.com/Melchiah04](https://github.com/Melchiah04)
