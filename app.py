"""Aplicación Streamlit: detección contextual de anomalías en la afluencia horaria de la Línea A.

Ejecutar desde la raíz del repositorio: ``streamlit run app.py``
"""
from __future__ import annotations

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from metro_anomaly import service as S
from metro_anomaly.data import SCORED_HOURS

ROOT = Path(__file__).resolve().parent
DEFAULT_DATE = pd.Timestamp("2023-08-28")
DIAS = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]

st.set_page_config(page_icon="🚆", page_title="Anomalías de afluencia — Línea A", layout="wide")


@st.cache_resource(show_spinner="Cargando histórico y modelo…")
def _resources():
    hist = S.load_history(ROOT / "data")
    det, thr, doc = S.load_model(ROOT / "artifacts" / "modelo.json")
    return hist, det, thr, doc


def _chart(r: S.DayResult):
    h = [c[1:] + "h" for c in SCORED_HOURS]
    df = pd.DataFrame({"Hora": h, "Observado": r.observed.to_numpy(), "Esperado": r.expected.to_numpy(),
                       "Banda inferior": r.band_low.to_numpy(), "Banda superior": r.band_high.to_numpy(),
                       "Fuera de banda": [c in r.hour_flags for c in SCORED_HOURS]})
    x = alt.X("Hora:N", sort=h, title="Hora")
    band = alt.Chart(df).mark_area(opacity=0.25).encode(x=x, y=alt.Y("Banda inferior:Q", title="Entradas por hora"), y2="Banda superior:Q")
    exp = alt.Chart(df).mark_line(strokeDash=[5, 3]).encode(x=x, y="Esperado:Q")
    obs = alt.Chart(df).mark_line(point=True).encode(x=x, y="Observado:Q", tooltip=["Hora", "Observado", "Esperado"])
    flags = alt.Chart(df[df["Fuera de banda"]]).mark_point(size=160, shape="diamond", filled=True, color="#d62728").encode(x=x, y="Observado:Q")
    return (band + exp + obs + flags).properties(height=340)


def _show_day(r: S.DayResult):
    head = f"**{r.date.date()}** · {DIAS[r.date.dayofweek]} · tipo de día: `{r.day_type}`" + (f" · 📅 {r.note}" if r.note else "")
    st.markdown(head)
    if r.alert:
        st.error(f"⚠️ ANOMALÍA — score {r.score:.2f} > umbral {r.threshold:.2f}. "
                 f"Desviación media del día: {r.deviation_pct:+.1f} %. Horas fuera de banda: "
                 f"{', '.join(c[1:] + 'h' for c in r.hour_flags) or 'ninguna (desplazamiento global del día)'}.")
    else:
        st.success(f"✅ Normal — score {r.score:.2f} ≤ umbral {r.threshold:.2f}. Desviación media del día: {r.deviation_pct:+.1f} %.")
    if r.missing_hours:
        st.warning(f"Calidad de datos: sin dato en {', '.join(c[1:] + 'h' for c in r.missing_hours)} (no se imputan; el día se evalúa con las horas disponibles).")
    st.altair_chart(_chart(r), use_container_width=True)
    with st.expander("Detalle por hora"):
        st.dataframe(pd.DataFrame({"Observado": r.observed.round(0), "Esperado": r.expected.round(0), "z": r.z.round(2),
                                   "Fuera de banda": [c in r.hour_flags for c in SCORED_HOURS]},
                                  index=[c[1:] + "h" for c in SCORED_HOURS]), use_container_width=True)


def main():
    st.title("🚆 Detección de anomalías en la afluencia — Línea A, Metro de Medellín")
    try:
        hist, det, thr, doc = _resources()
    except Exception as e:  # noqa: BLE001 — fallo de despliegue: mensaje claro, sin traza
        st.error(f"No se pudo cargar el histórico o el modelo: {type(e).__name__}: {e}")
        st.stop()

    with st.sidebar:
        st.header("Modelo")
        st.markdown(f"- Detector: `{doc['detector']}` (`{doc['agg']}`)\n- Umbral: **{thr:.3f}** (presupuesto de alertas del 5 % en DEV)\n"
                    f"- Entrenado con: {doc['trained_on'][0]} → {doc['trained_on'][1]}\n- Línea: {doc['line']}")
        m = doc.get("holdout_metrics", {})
        st.markdown("**Validación en holdout (2023-07 → 2024-06)**")
        st.markdown(f"- Tasa de alertas: {100 * m.get('tasa_alertas', float('nan')):.1f} %\n"
                    f"- Recall anomalías severas: {m.get('recall_severa', float('nan')):.2f}\n"
                    f"- Recall anomalías moderadas: {m.get('recall_moderada', float('nan')):.2f}\n"
                    f"- Alertas en festivos: {100 * m.get('tasa_alertas_festivos', float('nan')):.1f} %")
        st.caption("Limitaciones y criterios no cumplidos: docs/validacion/F4.md. La banda sombreada es el rango por hora "
                   "fuera del cual una hora se marca; la alerta del día usa el score agregado.")

    tab1, tab2 = st.tabs(["📆 Explorar histórico", "📤 Cargar archivo"])
    with tab1:
        valid = hist.index[hist[SCORED_HOURS].notna().any(axis=1)]
        d = st.date_input("Fecha", value=DEFAULT_DATE.date(), min_value=pd.Timestamp("2019-03-01").date(),
                          max_value=valid.max().date(), key="fecha")
        try:
            _show_day(S.score_days(det, thr, hist, [pd.Timestamp(d)])[0])
        except S.InputError as e:
            st.error(f"❌ {e}")

    with tab2:
        st.markdown("Suba un archivo **.xlsx** o **.csv** con el formato oficial (`Día`, `Línea de Servicio`, columnas "
                    "04:00:00…23:00:00). Incluya ≥ 4 semanas previas si las fechas son posteriores al histórico.")
        up = st.file_uploader("Archivo de afluencia", type=["xlsx", "csv"], key="archivo")
        if up is not None:
            try:
                new = S.parse_upload(up.getvalue(), up.name)
                ctx = S.merge_context(hist, new)
                days = [d for d in new.index if new.loc[d, SCORED_HOURS].notna().any()]
                results, errors = [], []
                for dd in days:
                    try:
                        results.extend(S.score_days(det, thr, ctx, [dd]))
                    except S.InputError as e:
                        errors.append(str(e))
                if errors:
                    st.warning(f"{len(errors)} día(s) no evaluables: " + "; ".join(errors[:3]))
                if results:
                    tbl = S.summary_table(results)
                    n_al = int((tbl["Estado"] == "ANOMALÍA").sum())
                    st.info(f"{len(results)} día(s) evaluados · {n_al} anomalía(s).")
                    st.dataframe(tbl, use_container_width=True, hide_index=True)
                    sel = st.selectbox("Ver detalle del día", [r.date.date() for r in results],
                                       index=next((i for i, r in enumerate(results) if r.alert), 0), key="detalle")
                    _show_day(next(r for r in results if r.date.date() == sel))
            except S.InputError as e:
                st.error(f"❌ {e}")


main()
