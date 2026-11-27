"""Investigación: cómo funcionan los modelos en uso y qué se aprendió de los autoencoders.

Página explicativa (rama de investigación). No alerta ni cambia ningún modelo: muestra, con datos precalculados, cómo
deciden el perfil contextual, la coherencia de red y el autoencoder estudiado, y el veredicto de cada hipótesis.
Datos: reports/investigacion_ae/tablero/*.json (scripts/investigacion_ae/) y artifacts/plataforma/horas.csv.gz.
"""
from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "reports" / "investigacion_ae" / "tablero"
ART = ROOT / "artifacts" / "plataforma"
st.set_page_config(page_icon="🔬", page_title="Investigación: cómo funcionan los modelos", layout="wide")

C_OBS, C_PR, C_AE, C_RED, C_COH = "#1f1e1d", "#2a78d6", "#eb6834", "#8a8883", "#1baf7a"
Y_FMT = "replace(format(datum.value, ',.0f'), ',', '.')"
TIPO = {"habil": "Hábil", "sabado": "Sábado", "domingo_festivo": "Domingo o festivo", "especial": "24 y 31 de diciembre"}


def num(x, d=0):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def pct(x, d=1):
    return "—" if x is None else f"{num(100 * x, d)} %"


def rows(name: str) -> pd.DataFrame:
    return pd.DataFrame(json.loads((TAB / f"{name}.json").read_text(encoding="utf-8"))["rows"])


@st.cache_data(show_spinner="Cargando resultados de la investigación…")
def load():
    data = {n: rows(n) for n in ("progreso", "ex_ejemplos", "ex_horas", "ex_latente", "ae5_resumen", "ae1_metricas")}
    h = pd.read_csv(ART / "horas.csv.gz", parse_dates=["fecha"])
    return data, h


def main():
    if not (TAB / "progreso.json").exists():
        st.error("Faltan los resultados de la investigación. Ejecute `python scripts/investigacion_ae/exportar_tablero.py`.")
        st.stop()
    d, horas = load()
    st.title("Cómo funcionan los modelos")
    st.caption("Página de investigación. Explica con datos reales cómo deciden los modelos en uso y qué se aprendió al estudiar "
               "autoencoders. No genera alertas ni cambia ningún modelo.")

    prog = d["progreso"]
    hyp = prog[prog["veredicto"] != ""]
    st.subheader("Qué modelos se usan hoy")
    st.markdown(
        "- **Perfil contextual** (todas las líneas): compara cada hora con lo típico para ese tipo de día y esa hora.\n"
        "- **Coherencia de red** (Líneas A, B y 1): descuenta lo que mueve a toda la red al mismo tiempo.\n"
        "- **Pronóstico del día siguiente** y **reglas de calendario y datos** (página Red multilínea).\n"
        f"- **Autoencoder**: estudiado como alternativa; de {len(hyp)} hipótesis, "
        f"{int((hyp['veredicto'] == 'cumple').sum())} se corroboró (solo descriptiva) y ninguna cambia un modelo de alerta.")
    with st.expander("Veredicto de cada hipótesis sobre el autoencoder"):
        st.table(prog[["id", "tarea", "detalle"]].rename(columns={"id": "Paso", "tarea": "Hipótesis o tarea", "detalle": "Resultado"}).set_index("Paso"))

    ej = d["ex_ejemplos"].sort_values("orden")
    labels = dict(zip(ej["ejemplo"], ej["titulo"]))
    sel = st.selectbox("Día de ejemplo", list(labels), format_func=lambda k: labels[k], key="ej")
    e = ej.set_index("ejemplo").loc[sel]
    H = d["ex_horas"][d["ex_horas"]["ejemplo"] == sel].sort_values("hora")
    st.caption(f"Tipo de dato: {e['tipo_dato']}.")

    # ---------------------------------------------------------------- 1. Perfil contextual
    st.header("1. Perfil contextual: ¿se parece a un día típico de su tipo?")
    c1, c2 = st.columns(2)
    with c1:
        L = H.melt(id_vars="hora", value_vars=["observado", "esperado_perfil"], var_name="serie", value_name="entradas")
        L["serie"] = L["serie"].map({"observado": "Observado", "esperado_perfil": "Esperado por el perfil"})
        st.altair_chart(alt.Chart(L).mark_line(point=True, strokeWidth=2).encode(
            x=alt.X("hora:O", title="Hora"), y=alt.Y("entradas:Q", title="Entradas por hora", axis=alt.Axis(labelExpr=Y_FMT)),
            color=alt.Color("serie:N", title=None, scale=alt.Scale(domain=["Observado", "Esperado por el perfil"], range=[C_OBS, C_PR]),
                            legend=alt.Legend(orient="top")),
            tooltip=[alt.Tooltip("hora:O", title="Hora"), alt.Tooltip("serie:N", title="Serie"), alt.Tooltip("entradas:Q", title="Entradas", format=",.0f")]
        ).properties(height=260), use_container_width=True)
    with c2:
        Z = H.assign(az=H["z_perfil"].abs())
        st.altair_chart(alt.Chart(Z).mark_bar(color=C_PR, cornerRadiusEnd=2).encode(
            x=alt.X("hora:O", title="Hora"), y=alt.Y("az:Q", title="|z| (desviación en unidades típicas)"),
            tooltip=[alt.Tooltip("hora:O", title="Hora"), alt.Tooltip("z_perfil:Q", title="z", format=".1f")]
        ).properties(height=260), use_container_width=True)
    st.markdown(
        f"**Cómo decide:** resta el nivel reciente de demanda, compara cada hora con la mediana de los días de su mismo tipo "
        f"y la divide por su variación típica (z). El score combina el pico de |z| y el desplazamiento de todo el día. "
        f"Este día: score {num(e['score_perfil'], 2)} frente a un umbral de {num(e['umbral_perfil'], 2)} → "
        f"**{'alerta' if e['alerta_perfil'] else 'sin alerta'}**.")

    # ---------------------------------------------------------------- 2. Coherencia de red
    st.header("2. Coherencia de red: ¿es propio de la línea o le pasa a toda la red?")
    cc1, cc2 = st.columns([1, 3])
    with cc1:
        ln = st.selectbox("Línea", ["LINEA A", "LINEA B", "LINEA 1"], format_func=lambda x: x.replace("LINEA ", "Línea "), key="coh_ln")
        fechas = sorted(horas.loc[horas["linea"] == ln, "fecha"].dt.date.unique())
        f = st.date_input("Fecha", value=pd.Timestamp("2023-08-28").date(), min_value=fechas[0], max_value=fechas[-1],
                          format="DD/MM/YYYY", key="coh_f")
    day = horas[horas["fecha"] == pd.Timestamp(f)]
    tgt = day[day["linea"] == ln].set_index("hora")
    if tgt.empty or tgt["z"].isna().all():
        cc2.info("No hay datos de esa línea en esa fecha.")
    else:
        oth = day[(day["linea"] != ln) & (day["observado"] > 0)]
        med = oth.groupby("hora")["z"].median()
        C = pd.DataFrame({"hora": tgt.index, "z de la línea": tgt["z"].to_numpy(), "mediana de la red": med.reindex(tgt.index).to_numpy(),
                          "propio de la línea": tgt["z_coh"].to_numpy()})
        C = C.melt(id_vars="hora", var_name="serie", value_name="z")
        dom = ["z de la línea", "mediana de la red", "propio de la línea"]
        cc2.altair_chart(alt.Chart(C).mark_line(point=True, strokeWidth=2).encode(
            x=alt.X("hora:O", title="Hora"), y=alt.Y("z:Q", title="z"),
            color=alt.Color("serie:N", title=None, scale=alt.Scale(domain=dom, range=[C_OBS, C_RED, C_COH]), legend=alt.Legend(orient="top")),
            strokeDash=alt.StrokeDash("serie:N", legend=None, scale=alt.Scale(domain=dom, range=[[1, 0], [5, 3], [1, 0]])),
            tooltip=[alt.Tooltip("hora:O", title="Hora"), alt.Tooltip("serie:N", title="Serie"), alt.Tooltip("z:Q", format=".1f")]
        ).properties(height=280), use_container_width=True)
    st.markdown("**Cómo decide:** toma el z de la línea y le resta lo que explica la mediana de las otras 10 líneas en esa misma "
                "hora (ajustado hora por hora). Si toda la red cae por lluvia o un festivo, lo propio de la línea queda cerca de 0; "
                "si solo cae la línea, se mantiene alto. En la Línea A mejoró la detección moderada de 0,70 a 0,90.")

    # ---------------------------------------------------------------- 3. Autoencoder (investigación)
    st.header("3. Autoencoder (estudiado, no en uso): ¿puede reconstruir el día?")
    st.markdown("Recibe las 19 horas sin el nivel de demanda y 11 rasgos de calendario, las **comprime en 4 números** y "
                "reconstruye las 19 horas. Lo que no logra reconstruir delata la anomalía; el score es el promedio de las 2 horas "
                "más desviadas.")
    a1, a2 = st.columns(2)
    with a1:
        L = H.melt(id_vars="hora", value_vars=["observado", "esperado_ae", "esperado_perfil"], var_name="serie", value_name="entradas")
        nm = {"observado": "Observado", "esperado_ae": "Autoencoder (reconstruido)", "esperado_perfil": "Perfil (esperado)"}
        L["serie"] = L["serie"].map(nm)
        st.altair_chart(alt.Chart(L).mark_line(point=True, strokeWidth=2).encode(
            x=alt.X("hora:O", title="Hora"), y=alt.Y("entradas:Q", title="Entradas por hora", axis=alt.Axis(labelExpr=Y_FMT)),
            color=alt.Color("serie:N", title=None, scale=alt.Scale(domain=list(nm.values()), range=[C_OBS, C_AE, C_PR]), legend=alt.Legend(orient="top", labelLimit=260)),
            tooltip=[alt.Tooltip("hora:O", title="Hora"), alt.Tooltip("serie:N", title="Serie"), alt.Tooltip("entradas:Q", format=",.0f")]
        ).properties(height=280), use_container_width=True)
    with a2:
        Z = H.assign(az=H["z_ae"].abs(), define=np.where(H["top2"], "Define el score", "Otras horas"))
        st.altair_chart(alt.Chart(Z).mark_bar(cornerRadiusEnd=2).encode(
            x=alt.X("hora:O", title="Hora"), y=alt.Y("az:Q", title="|z| del autoencoder", scale=alt.Scale(type="symlog")),
            color=alt.Color("define:N", title=None, scale=alt.Scale(domain=["Define el score", "Otras horas"], range=["#d03b3b", C_AE]), legend=alt.Legend(orient="top")),
            tooltip=[alt.Tooltip("hora:O", title="Hora"), alt.Tooltip("z_ae:Q", title="z", format=".1f")]
        ).properties(height=280), use_container_width=True)
    st.markdown(f"Este día: score del autoencoder {num(e['score_ae'], 2)} frente a un umbral de {num(e['umbral_ae'], 2)} → "
                f"**{'alerta' if e['alerta_ae'] else 'sin alerta'}**"
                + (f", horas {', '.join(str(x) for x in e['horas_top2'])}." if e["alerta_ae"] else "."))
    st.info("Lo que mostró la investigación: el autoencoder ve mejor las rupturas de forma (caídas de pocas horas, horas duplicadas), "
            "pero casi no ve los cambios de nivel (caídas de todo el día) porque trabaja sin el nivel. Con un día muy anómalo, "
            "la reconstrucción entera se deforma y puede culpar a horas normales (ver el incidente del 28/08/2023).")

    st.subheader("Lo que sí se corroboró: el resumen de 4 números organiza los días")
    lat = d["ex_latente"].copy()
    lat["tipo"] = lat["tipo_dia"].map(TIPO).fillna(lat["tipo_dia"])
    lat["supera"] = np.where(lat["score_rel"] > 1, "Supera el umbral", "No supera")
    l1, l2 = st.columns([3, 2])
    with l1:
        st.altair_chart(alt.Chart(lat).mark_circle(size=28, opacity=0.75).encode(
            x=alt.X("pc1:Q", title="Componente 1", scale=alt.Scale(type="symlog")),
            y=alt.Y("pc2:Q", title="Componente 2", scale=alt.Scale(type="symlog")),
            color=alt.Color("tipo:N", title="Tipo de día", legend=alt.Legend(orient="top")),
            tooltip=[alt.Tooltip("fecha:N", title="Fecha"), alt.Tooltip("tipo:N", title="Tipo"), alt.Tooltip("score_rel:Q", title="Score/umbral", format=".2f"),
                     alt.Tooltip("nota:N", title="Calendario")]
        ).properties(height=320), use_container_width=True)
    with l2:
        r = d["ae5_resumen"].sort_values("orden")
        a = r.iloc[0]
        st.metric("Separación de días con eventos verificados (AUC)", num(a["auc"], 3), help="1 = separa perfectamente; 0,5 = al azar. Exigido: 0,85.")
        st.caption(f"Intervalo 95 %: {num(a['ic_inf'], 2)} a {num(a['ic_sup'], 2)}, con solo 9 días con eventos. Separa las jornadas "
                   "electorales sin cobro y los cierres anticipados de diciembre, pero no las interrupciones parciales. "
                   "Uso: explicar, no alertar.")
    st.caption("Fuentes: docs/investigacion_autoencoders/INFORME.md, reports/investigacion_ae/ y artifacts/plataforma/. "
               "Los modelos de cada ejemplo son los de la evaluación mes a mes (solo datos anteriores al mes del día mostrado).")


main()
