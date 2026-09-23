"""
wfm_utils.py

Funciones de planeacion de personal (Workforce Management) que traducen un
volumen pronosticado de contactos en la dotacion (FTE) necesaria, usando el
modelo Erlang C -- el estandar de la industria de centros de contacto para
calcular cuantos agentes se necesitan para alcanzar un Nivel de Servicio (NS)
objetivo (ej. "80% de las llamadas contestadas en <= 20 segundos").

Se usa tanto en 04_calculo_dotacion.py como en la app de Streamlit (05_app_streamlit.py),
para que el mismo calculo se pueda explorar de forma interactiva por escenario.
"""

import math


def erlang_c_prob_espera(agentes: int, intensidad_erlangs: float) -> float:
    """Probabilidad de que un contacto tenga que esperar (formula de Erlang C)."""
    if agentes <= intensidad_erlangs:
        return 1.0  # sistema inestable: mas trafico del que los agentes pueden atender
    suma = sum((intensidad_erlangs ** n) / math.factorial(n) for n in range(agentes))
    ultimo_termino = (intensidad_erlangs ** agentes) / math.factorial(agentes)
    factor_ocupacion = agentes / (agentes - intensidad_erlangs)
    numerador = ultimo_termino * factor_ocupacion
    return numerador / (suma + numerador)


def nivel_de_servicio(agentes: int, intensidad_erlangs: float, aht_seg: float,
                       objetivo_seg: float) -> float:
    """NS = probabilidad de que un contacto sea atendido dentro de `objetivo_seg`."""
    if agentes <= intensidad_erlangs:
        return 0.0
    p_espera = erlang_c_prob_espera(agentes, intensidad_erlangs)
    exponente = -(agentes - intensidad_erlangs) * (objetivo_seg / aht_seg)
    return 1 - p_espera * math.exp(exponente)


def agentes_requeridos_erlang_c(contactos_hora: float, aht_seg: float,
                                 ns_objetivo: float, tiempo_objetivo_seg: float,
                                 max_agentes: int = 400) -> int:
    """
    Numero minimo de agentes 'en base' (sin shrinkage) que cumple el NS objetivo,
    buscando el minimo m tal que nivel_de_servicio(m) >= ns_objetivo.
    """
    intensidad = (contactos_hora * aht_seg) / 3600.0  # trafico en Erlangs
    agentes = max(1, math.ceil(intensidad))
    while agentes < max_agentes:
        if nivel_de_servicio(agentes, intensidad, aht_seg, tiempo_objetivo_seg) >= ns_objetivo:
            return agentes
        agentes += 1
    return max_agentes


def dotacion_diaria(volumen_dia: float, horas_operacion: float, aht_seg: float,
                     ns_objetivo: float, tiempo_objetivo_seg: float,
                     shrinkage: float) -> dict:
    """
    A partir del volumen total del dia, calcula:
      - contactos_hora: volumen distribuido uniformemente en las horas de operacion
        (simplificacion razonable para un ejercicio de portafolio; en una operacion
        real esto se haria por intervalos de 30 min con su propio perfil de trafico).
      - agentes_base: agentes necesarios para el NS objetivo (Erlang C).
      - fte_con_shrinkage: agentes base ajustados por shrinkage (ausentismo, pausas,
        formacion, tiempo administrativo) para llegar a la dotacion real necesaria.
    """
    contactos_hora = volumen_dia / horas_operacion
    agentes_base = agentes_requeridos_erlang_c(
        contactos_hora, aht_seg, ns_objetivo, tiempo_objetivo_seg
    )
    fte_con_shrinkage = math.ceil(agentes_base / (1 - shrinkage))
    return {
        "contactos_hora": round(contactos_hora, 1),
        "agentes_base": agentes_base,
        "fte_con_shrinkage": fte_con_shrinkage,
    }
