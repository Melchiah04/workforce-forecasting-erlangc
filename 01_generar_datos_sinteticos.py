"""
01_generar_datos_sinteticos.py

IMPORTANTE - TRANSPARENCIA SOBRE LOS DATOS:
Este proyecto usa una serie de tiempo SINTETICA (simulada), generada con este
script, NO datos reales de ninguna empresa. Se construye para que se comporte
como una serie real de volumen de contactos de un centro de servicio (tendencia,
estacionalidad semanal, estacionalidad anual, feriados y ruido aleatorio), con el
fin de demostrar de forma honesta el flujo completo de forecasting y planeacion
de dotacion (Workforce Management) sobre una base reproducible y compartible
publicamente (datos reales de una operacion nunca podrian publicarse en GitHub).

Genera:
    data/volumen_contactos.csv  -> serie diaria de volumen de contactos (2023-2025)
"""

import numpy as np
import pandas as pd
from pathlib import Path

OUT_DIR = Path(__file__).parent / "data"
OUT_DIR.mkdir(exist_ok=True)

np.random.seed(42)

# --- Rango de fechas: 3 años de historico diario ---
start = "2023-01-01"
end = "2025-12-31"
dates = pd.date_range(start=start, end=end, freq="D")
n = len(dates)
t = np.arange(n)

# --- Componente de tendencia: crecimiento gradual de la operacion ---
tendencia = 1400 + 0.35 * t

# --- Estacionalidad semanal: lunes y martes son los dias de mayor volumen,
#     domingo el de menor (patron tipico de servicio al cliente) ---
dow_factor = {
    0: 1.18,  # lunes
    1: 1.10,  # martes
    2: 1.00,  # miercoles
    3: 0.97,  # jueves
    4: 0.95,  # viernes
    5: 0.70,  # sabado
    6: 0.55,  # domingo
}
estacionalidad_semanal = np.array([dow_factor[d.weekday()] for d in dates])

# --- Estacionalidad anual: pico en enero (renovaciones/inicio de ano) y
#     noviembre-diciembre (temporada alta), valle a mitad de ano ---
dia_del_anio = np.array([d.dayofyear for d in dates])
estacionalidad_anual = 1 + 0.18 * np.sin(2 * np.pi * (dia_del_anio - 15) / 365.25) \
                          + 0.10 * np.sin(4 * np.pi * (dia_del_anio - 320) / 365.25)

# --- Feriados colombianos aproximados (reduccion de volumen operativo) ---
feriados = pd.to_datetime([
    "2023-01-01", "2023-01-09", "2023-03-20", "2023-04-06", "2023-04-07",
    "2023-05-01", "2023-05-22", "2023-06-12", "2023-06-19", "2023-07-03",
    "2023-07-20", "2023-08-07", "2023-08-21", "2023-10-16", "2023-11-06",
    "2023-11-13", "2023-12-08", "2023-12-25",
    "2024-01-01", "2024-01-08", "2024-03-25", "2024-03-28", "2024-03-29",
    "2024-05-01", "2024-05-13", "2024-06-03", "2024-06-10", "2024-07-01",
    "2024-07-20", "2024-08-07", "2024-08-19", "2024-10-14", "2024-11-04",
    "2024-11-11", "2024-12-08", "2024-12-25",
    "2025-01-01", "2025-01-06", "2025-03-24", "2025-04-17", "2025-04-18",
    "2025-05-01", "2025-06-02", "2025-06-23", "2025-06-30", "2025-07-20",
    "2025-08-07", "2025-08-18", "2025-10-13", "2025-11-03", "2025-11-17",
    "2025-12-08", "2025-12-25",
])
factor_feriado = np.array([0.45 if d in feriados else 1.0 for d in dates])

# --- Ruido aleatorio (variabilidad diaria propia de cualquier operacion) ---
ruido = np.random.normal(loc=1.0, scale=0.05, size=n)

# --- Serie final ---
volumen = tendencia * estacionalidad_semanal * estacionalidad_anual * factor_feriado * ruido
volumen = np.round(np.clip(volumen, 200, None)).astype(int)

df = pd.DataFrame({"fecha": dates, "volumen_contactos": volumen})
df.to_csv(OUT_DIR / "volumen_contactos.csv", index=False)

print(f"Serie generada: {n} dias, desde {start} hasta {end}")
print(df.describe())
print(f"Guardado en: {OUT_DIR / 'volumen_contactos.csv'}")
