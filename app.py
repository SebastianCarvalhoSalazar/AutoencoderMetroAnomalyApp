"""Aplicación Streamlit: detección de anomalías estadísticas en la afluencia horaria de la Línea A.

Ejecutar desde la raíz del repositorio: ``streamlit run app.py``
La app NO confirma incidentes: señala desviaciones estadísticas que deben verificarse con registros operativos.
"""
from __future__ import annotations

from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from metro_anomaly import service as S
from metro_anomaly.data import SCORED_HOURS

ROOT = Path(__file__).resolve().parent
DEFAULT_DATE = pd.Timestamp("2023-08-28")
MIN_DATE = pd.Timestamp("2019-01-29")  # primer día con ≥ 28 días de historia previa (nivel causal)
DIAS = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
SERIES = ["Observado", "Esperado", "Rango normal", "Fuera de rango", "Sin dato"]
COLORS = ["#1f4e9a", "#4b5563", "#9ec5fe", "#b42318", "#6b7280"]
Y_FMT = "replace(format(datum.value, ',.0f'), ',', '.')"
SEV_TXT = {"moderada": "moderada", "alta": "alta", "extrema": "extrema"}

st.set_page_config(page_icon="🚆", page_title="Anomalías de afluencia — Línea A", layout="wide")


def num(x: float, d: int = 0) -> str:
    """Formato numérico colombiano: miles con punto, decimales con coma."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:,.{d}f}".replace(",", "§").replace(".", ",").replace("§", ".")


def fecha(d) -> str:
    return pd.Timestamp(d).strftime("%d/%m/%Y")


@st.cache_resource(show_spinner="Cargando histórico y modelo…")
def _resources():
    hist = S.load_history(ROOT / "data")
    det, thr, doc = S.load_model(ROOT / "artifacts" / "modelo.json")
    return hist, det, thr, doc, S.reference_scores(doc)


# ------------------------------------------------------------------------------------------- gráficas
def hourly_chart(r: S.DayResult):
    h = [c[1:] for c in SCORED_HOURS]
    obs, exp = r.observed.to_numpy(float), r.expected.to_numpy(float)
    top = float(np.nanmax(np.r_[obs, exp])) * 1.6
    clipped = bool(np.nanmax(r.band_high.to_numpy(float)) > top)
    base = pd.DataFrame({"Hora": h, "Observado": obs, "Esperado": exp, "lo": r.band_low.to_numpy(float),
                         "hi": np.minimum(r.band_high.to_numpy(float), top),
                         "fuera": [c in r.hour_flags for c in SCORED_HOURS], "sin_dato": np.isnan(obs)})
    x = alt.X("Hora:N", sort=h, title="Hora del día (inicio de la franja horaria)", axis=alt.Axis(labelAngle=0))
    y_scale = alt.Scale(domain=[0, top], nice=False)
    color = alt.Color("Serie:N", scale=alt.Scale(domain=SERIES, range=COLORS), legend=alt.Legend(title=None, orient="top", columns=2))
    band = alt.Chart(base).transform_calculate(Serie="'Rango normal'").mark_area(opacity=0.45, clip=True).encode(
        x=x, y=alt.Y("lo:Q", scale=y_scale, title="Entradas por hora (pasajeros)", axis=alt.Axis(labelExpr=Y_FMT)),
        y2="hi:Q", color=color)
    lines = alt.Chart(base).transform_fold(["Observado", "Esperado"], as_=["Serie", "valor"]).mark_line(point=True, clip=True).encode(
        x=x, y=alt.Y("valor:Q", scale=y_scale), color=color,
        strokeDash=alt.StrokeDash("Serie:N", scale=alt.Scale(domain=["Observado", "Esperado"], range=[[1, 0], [6, 4]]), legend=None),
        tooltip=[alt.Tooltip("Hora:N"), alt.Tooltip("Serie:N"), alt.Tooltip("valor:Q", format=",.0f", title="Entradas")])
    flags = alt.Chart(base[base["fuera"]]).transform_calculate(Serie="'Fuera de rango'").mark_point(
        size=170, shape="diamond", filled=True, clip=True).encode(x=x, y=alt.Y("Observado:Q", scale=y_scale), color=color)
    miss = alt.Chart(base[base["sin_dato"]].assign(cero=0)).transform_calculate(Serie="'Sin dato'").mark_point(
        size=150, shape="triangle-up", filled=True).encode(x=x, y=alt.Y("cero:Q", scale=y_scale), color=color,
                                                           tooltip=[alt.Tooltip("Hora:N"), alt.Tooltip("Serie:N")])
    return (band + lines + flags + miss).properties(height=360), clipped


def timeline_chart(ts: pd.DataFrame, thr: float, selected=None):
    df = ts.reset_index().rename(columns={"index": "Fecha"})
    df["Fecha"] = pd.to_datetime(df["Fecha"])
    df["Estado"] = np.where(df["alerta"], "Anomalía estadística", "Sin anomalía")
    df["Fecha_txt"] = df["Fecha"].dt.strftime("%d/%m/%Y")
    x = alt.X("Fecha:T", title="Fecha", axis=alt.Axis(format="%d/%m", labelAngle=0))
    y_axis = alt.Axis(values=[0, 1, 2, 5, 10, 20, 50, 100], labelExpr="replace(format(datum.value, ',.0f'), ',', '.')")
    y_scale = alt.Scale(type="symlog")
    y = alt.Y("score:Q", title="Score (escala log.)", scale=y_scale, axis=y_axis)
    line = alt.Chart(df).mark_line(color="#9ca3af").encode(x=x, y=y)
    pts = alt.Chart(df).mark_circle(size=60).encode(
        x=x, y=y, color=alt.Color("Estado:N", scale=alt.Scale(domain=["Sin anomalía", "Anomalía estadística"], range=["#4b5563", "#b42318"]),
                                  legend=alt.Legend(title=None, orient="top")),
        tooltip=[alt.Tooltip("Fecha_txt:N", title="Fecha"), alt.Tooltip("score:Q", format=".2f", title="Score"), "Estado:N"])
    # Todas las capas deben declarar el mismo eje: si una usa el eje por defecto, al fusionarse se pierden las
    # marcas configuradas (solo se veían 0 y 50); con axis=None desaparece el eje completo.
    y_rule = alt.Y("u:Q", title="Score (escala log.)", scale=y_scale, axis=y_axis)
    rule = alt.Chart(pd.DataFrame({"u": [thr], "t": [f"Umbral {num(thr, 2)}"]})).mark_rule(strokeDash=[4, 3], color="#b42318").encode(y=y_rule)
    lab = alt.Chart(pd.DataFrame({"u": [thr], "t": [f"Umbral {num(thr, 2)}"]})).mark_text(align="left", dx=4, dy=-6, color="#b42318").encode(
        y=y_rule, x=alt.value(0), text="t:N")
    layers = [line, rule, lab, pts]
    if selected is not None and pd.Timestamp(selected) in set(df["Fecha"]):
        sel = df[df["Fecha"] == pd.Timestamp(selected)]
        layers.append(alt.Chart(sel).mark_point(size=260, shape="circle", filled=False, strokeWidth=2.5, color="#111827").encode(x=x, y=y))
    return alt.layer(*layers).properties(height=220)


# ------------------------------------------------------------------------------------------- vistas
def status_box(r: S.DayResult):
    sev = f" · severidad **{SEV_TXT[r.severity]}**" if r.severity else ""
    if r.category == "efecto_de_medicion":
        st.info(f"🗳️ **Efecto de medición conocido** — {S.ELECTION_FARE_FREE_NOTE}. La desviación (score {num(r.score, 2)}) "
                "no debe interpretarse como un incidente ni como una caída de afluencia.")
    elif r.category == "posible_problema_de_datos":
        st.warning("🛠️ **Posible problema de datos** — " + "; ".join(f for f in r.dq_flags if not f.startswith("Sin dato"))
                   + ". Antes de interpretar la afluencia, verifique la captura/registro de los datos.")
    elif r.category == "anomalia_estadistica":
        horas = ", ".join(c[1:] + " h" for c in r.hour_flags) or "ninguna hora aislada (desplazamiento de todo el día)"
        st.warning(f"⚠️ **Anomalía estadística detectada**{sev}. La afluencia se aparta de lo esperado para este contexto. "
                   f"Horas fuera de rango: {horas}. **No es un incidente confirmado**: requiere verificación con los "
                   "registros operativos del Metro (Centro de Control) antes de cualquier decisión.")
    else:
        st.success("✅ **Sin anomalía estadística** — la afluencia está dentro de lo esperado para este contexto. "
                   "Esto no descarta incidentes que no alteren las entradas de la línea (p. ej., interrupciones parciales).")
    if r.note:
        st.info(f"📅 Contexto de calendario conocido: **{r.note}**. Una desviación puede explicarse por este evento.")
    miss = [f for f in r.dq_flags if f.startswith("Sin dato")]
    # (las banderas de jornada electoral ya se muestran en el recuadro principal)
    if miss:
        st.info("ℹ️ Calidad de datos: " + "; ".join(miss) + ". No se imputan; el día se evalúa con las horas disponibles.")


def show_day(r: S.DayResult, ts: pd.DataFrame | None, thr: float, key: str):
    st.subheader(f"{DIAS[r.date.dayofweek].capitalize()} {fecha(r.date)} · {r.day_type_label}", anchor=False)
    status_box(r)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Score del día", num(r.score, 2), help="Desviación agregada respecto de lo esperado. Hay alerta si supera el umbral.")
    c2.metric("Umbral", num(thr, 2), help="Cuantil 95 % de los scores de días reales de referencia (2022-04 → 2023-06).")
    c3.metric("Frecuencia histórica", "—" if r.exceedance is None else f"{num(100 * r.exceedance, 1)} %",
              help="Porcentaje de días de referencia con un score igual o mayor. No es la probabilidad de un incidente.")
    c4.metric("Desviación media del día", f"{'+' if r.deviation_pct >= 0 else ''}{num(r.deviation_pct, 1)} %",
              help="Diferencia media (geométrica) entre las entradas observadas y las esperadas.")
    chart, clipped = hourly_chart(r)
    st.markdown("**Entradas por hora: observado frente a lo esperado**")
    st.altair_chart(chart, use_container_width=True)
    if clipped:
        st.markdown("*Nota: la banda del rango normal se recorta en la parte superior del gráfico para conservar la "
                    "legibilidad; los valores completos están en la tabla de detalle.*")
    if ts is not None and len(ts):
        st.markdown(f"**Evolución del score diario** ({fecha(ts.index.min())} – {fecha(ts.index.max())})")
        st.altair_chart(timeline_chart(ts, thr, r.date), use_container_width=True)
    with st.expander("Detalle por hora (tabla)"):
        st.table(pd.DataFrame({
            "Hora": [c[1:] + " h" for c in SCORED_HOURS],
            "Observado": [num(v) for v in r.observed], "Esperado": [num(v) for v in r.expected],
            "Rango normal": [f"{num(a)} – {num(b)}" for a, b in zip(r.band_low, r.band_high)],
            "z": [num(v, 2) for v in r.z], "Fuera de rango": ["sí" if c in r.hour_flags else "" for c in SCORED_HOURS],
        }).set_index("Hora"))


def sidebar(doc, thr):
    with st.sidebar:
        st.header("Acerca del modelo", anchor=False)
        m = doc.get("holdout_metrics", {})
        st.markdown(
            f"- **Método:** perfil esperado por tipo de día y hora, con nivel reciente (sin datos futuros).\n"
            f"- **Umbral:** {num(thr, 2)} (≈ 5 % de días de referencia con alerta).\n"
            f"- **Entrenado con:** {fecha(doc['trained_on'][0])} – {fecha(doc['trained_on'][1])}.\n- **Línea:** A.")
        st.markdown("**Validación (2023-07 → 2024-06)**")
        st.markdown(
            f"- Días con alerta: {num(100 * m.get('tasa_alertas', np.nan), 1)} %\n"
            f"- Anomalías simuladas severas detectadas: {num(100 * m.get('recall_severa', np.nan), 0)} %\n"
            f"- Anomalías simuladas moderadas detectadas: {num(100 * m.get('recall_moderada', np.nan), 0)} %\n"
            f"- Alertas en festivos: {num(100 * m.get('tasa_alertas_festivos', np.nan), 1)} %")
        st.markdown("**Limitaciones importantes**")
        st.markdown(
            "- Una alerta es una **anomalía estadística**, no un incidente confirmado.\n"
            "- De 3 incidentes reales verificados, solo 1 (suspensión total en hora pico) dejó señal en estos datos; "
            "las interrupciones parciales pueden pasar desapercibidas.\n"
            "- No anticipa eventos: detecta desviaciones después de que ocurren (con datos por hora, ~1 h después).\n"
            "- Datos agregados por línea: no localiza estaciones ni causas.")


def tab_history(hist, det, thr, ref):
    valid = hist.index[hist[SCORED_HOURS].notna().any(axis=1)]
    d = st.date_input("Fecha a analizar", value=DEFAULT_DATE.date(), min_value=MIN_DATE.date(),
                      max_value=valid.max().date(), format="DD/MM/YYYY", key="fecha")
    d = pd.Timestamp(d)
    if d < MIN_DATE or d > valid.max():
        st.error(f"❌ La fecha {fecha(d)} está fuera del rango con datos ({fecha(MIN_DATE)} – {fecha(valid.max())}).")
        return
    try:
        r = S.score_days(det, thr, hist, [d], ref)[0]
    except S.InputError as e:
        st.error(f"❌ {e}")
        return
    ts = S.score_series(det, thr, hist, d - pd.Timedelta(days=59), d)
    show_day(r, ts, thr, "hist")


def tab_upload(hist, det, thr, ref):
    st.markdown("Suba un archivo **.xlsx** o **.csv** con el formato oficial de afluencia (columnas «Día», "
                "«Línea de Servicio» y 04:00:00 … 23:00:00; máximo 20 MB). Se evalúa la **Línea A**. Si las fechas son "
                "posteriores al histórico, incluya al menos 4 semanas previas en el mismo archivo.")
    up = st.file_uploader("Archivo de afluencia", type=["xlsx", "csv"], key="archivo")
    if up is None:
        st.info("Aún no se ha cargado ningún archivo.")
        return
    try:
        with st.spinner(f"Validando y evaluando «{up.name}»…"):
            new = S.parse_upload(up.getvalue(), up.name)
            ctx = S.merge_context(hist, new)
            results, errors = [], []
            for dd in [d for d in new.index if new.loc[d, SCORED_HOURS].notna().any()]:
                try:
                    results.extend(S.score_days(det, thr, ctx, [dd], ref))
                except S.InputError as e:
                    errors.append(str(e))
    except S.InputError as e:
        st.error(f"❌ No se pudo procesar «{up.name}»: {e}")
        return
    st.subheader(f"Resultados de «{up.name}»", anchor=False)
    if errors:
        st.warning(f"⚠️ {len(errors)} día(s) no evaluables: " + "; ".join(errors[:3]))
    if not results:
        st.info("No hay días evaluables en el archivo.")
        return
    tbl = S.summary_table(results)
    n_an = int((tbl["Estado"] == S.CATEGORY_LABELS["anomalia_estadistica"]).sum())
    n_dq = int(tbl["Estado"].isin([S.CATEGORY_LABELS["posible_problema_de_datos"], S.CATEGORY_LABELS["efecto_de_medicion"]]).sum())
    n_miss = sum(any(f.startswith("Sin dato") for f in r.dq_flags) for r in results)
    st.info(f"{len(results)} día(s) evaluados · {n_an} anomalía(s) estadística(s) · {n_dq} posible(s) problema(s) de datos o de medición · "
            f"{n_miss} día(s) con horas sin dato.")
    ts = pd.DataFrame({"score": [r.score for r in results], "alerta": [r.alert for r in results]},
                      index=pd.DatetimeIndex([r.date for r in results]))
    st.dataframe(tbl, use_container_width=True, hide_index=True)
    st.download_button("⬇️ Descargar resultados (CSV)", tbl.to_csv(index=False).encode("utf-8"),
                       file_name=f"resultados_{Path(up.name).stem}.csv", mime="text/csv", key="descarga")
    order = sorted(range(len(results)), key=lambda i: (results[i].category == "sin_anomalia", results[i].date))
    labels = {i: f"{fecha(results[i].date)} — {results[i].status}" for i in order}
    sel = st.selectbox("Ver detalle del día", order, format_func=labels.get, key="detalle")
    show_day(results[sel], ts, thr, "up")


def main():
    st.title("🚆 Anomalías de afluencia — Línea A del Metro de Medellín", anchor=False)
    st.markdown("Detecta **desviaciones estadísticas** de las entradas por hora respecto de lo esperado para el contexto "
                "(día de la semana, festivos, temporada y nivel reciente). Una alerta indica que algo merece revisión; "
                "**no confirma un incidente**.")
    try:
        hist, det, thr, doc, ref = _resources()
    except Exception as e:  # noqa: BLE001 — fallo de despliegue: mensaje claro, sin traza
        st.error(f"❌ No se pudo cargar el histórico o el modelo ({type(e).__name__}). Contacte al administrador.")
        st.stop()
    sidebar(doc, thr)
    tab1, tab2 = st.tabs(["📆 Explorar histórico", "📤 Cargar archivo"])
    with tab1:
        tab_history(hist, det, thr, ref)
    with tab2:
        tab_upload(hist, det, thr, ref)


main()
