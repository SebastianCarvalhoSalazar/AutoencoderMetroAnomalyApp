"""Plataforma multilínea (F10.5): red, desviaciones por línea y hora, evaluación del día, pronóstico y catálogo.

Lee artefactos precalculados por `scripts/fase3/build_plataforma.py` (reproducción cronológica causal).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "plataforma"
st.set_page_config(page_icon="🚆", page_title="Red multilínea — Metro de Medellín", layout="wide")

# Paleta divergente (validada en la guía de visualización): caída = rojo, aumento = azul, neutro gris.
DIV = ["#b3261e", "#e34948", "#f3a6a0", "#f0efec", "#9cc3f0", "#2a78d6", "#174f94"]
Z_CLIP = 8
Y_FMT = "replace(format(datum.value, ',.0f'), ',', '.')"  # miles con punto (es-CO)
KIND = {"real": ("Dato real", "#0b6e4f"), "estimacion": ("Estimación", "#5b4bb7"),
        "simulacion": ("Simulación", "#9a5b00"), "prediccion": ("Predicción", "#1f4e9a")}
CAT_ICON = {"anomalia_local": "🔴", "anomalia_multilinea": "🟠", "anomalia_generalizada": "🟣", "problema_de_datos": "⚠️",
            "ausencia_simultanea_red": "⛔", "efecto_de_medicion": "🗳️", "sin_anomalia": "✅", "no_evaluable": "◻️", "sin_datos": "—"}
# Posiciones del esquema de relaciones entre líneas (NO geográfico): nodos = líneas, aristas = estaciones de integración.
POS = {"LINEA A": (0, 0), "LINEA B": (-2, 0), "LINEA J": (-4, 0.4), "LINEA T-A": (2, -0.6), "LINEA H": (4, -0.2),
       "LINEA M": (4, -1.4), "LINEA K": (0.9, 2.0), "LINEA P": (-0.9, 2.0), "LINEA L": (1.8, 3.2), "LINEA O": (-2.2, 1.6),
       "LINEA 1": (2.4, 1.3), "LINEA 2": (3.6, 2.0)}


def num(x, d=0):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def badge(kind: str, extra: str = ""):
    label, color = KIND[kind]
    st.markdown(f"<span style='background:{color};color:#fff;border-radius:6px;padding:2px 8px;font-size:0.85rem;"
                f"font-weight:600'>{label}</span> <span style='font-size:0.9rem'>{extra}</span>", unsafe_allow_html=True)


@st.cache_data(show_spinner="Cargando artefactos de la plataforma…")
def load():
    man = json.loads((ART / "manifiesto.json").read_text(encoding="utf-8"))
    horas = pd.read_csv(ART / "horas.csv.gz", parse_dates=["fecha"])
    dias = pd.read_csv(ART / "dias.csv.gz", parse_dates=["fecha"], dtype={"horas_fuera": str, "lineas_acompanan": str,
                                                                          "lineas_sentido_opuesto": str}).fillna({"lineas_acompanan": "", "lineas_sentido_opuesto": "",
                                                                            "horas_fuera": "", "nota_calendario": "", "alcance": ""})
    fc = pd.read_csv(ART / "pronostico_holdout.csv.gz", parse_dates=["fecha"])
    nxt = pd.read_csv(ART / "pronostico_siguiente.csv", parse_dates=["fecha"])
    lineas = pd.read_csv(ROOT / "data" / "lineas.csv")
    topo = pd.read_csv(ROOT / "data" / "topologia_red.csv")
    return man, horas, dias, fc, nxt, lineas, topo


def short(ln: str) -> str:
    return ln.replace("LINEA ", "")


def detalle(horas: str, acomp: str, opuesto: str) -> str:
    """Una sola columna de detalle (la tabla debe caber en 390 px de ancho)."""
    def lst(x):
        return ", ".join(short(y) for y in x.split(",") if y)
    parts = []
    if horas:
        parts.append(f"Horas con |z| > 3: {horas.replace(',', ', ')}")
    if acomp:
        parts.append(f"Acompañan: {lst(acomp)}")
    if opuesto:
        parts.append(f"Sentido opuesto: {lst(opuesto)}")
    return " · ".join(parts) or "—"


def network_chart(h_day: pd.DataFrame, lineas: pd.DataFrame, topo: pd.DataFrame, modeled: set, hour_fixed: int | None = None):
    """Esquema de la red. Sin ``hour_fixed``: control deslizante en el navegador. Con ``hour_fixed``: fotograma de la animación."""
    nodes = pd.DataFrame([{"linea": ln, "x": x, "y": y} for ln, (x, y) in POS.items()])
    nodes = nodes.merge(lineas[["linea", "modo"]], on="linea", how="left")
    rows = []
    for _, n in nodes.iterrows():
        for h in range(4, 23):
            g = h_day[(h_day["linea"] == n["linea"]) & (h_day["hora"] == h)]
            z = float(g["z"].iloc[0]) if len(g) and pd.notna(g["z"].iloc[0]) else np.nan
            obs = float(g["observado"].iloc[0]) if len(g) else np.nan
            esp = float(g["esperado"].iloc[0]) if len(g) else np.nan
            rows.append({**n, "hora": h, "z": z, "zc": np.clip(z, -Z_CLIP, Z_CLIP) if not np.isnan(z) else np.nan,
                         "observado": obs, "esperado": esp, "etiqueta": short(n["linea"]),
                         "estado": "sin modelar (no opera todas las horas)" if n["linea"] not in modeled else
                         ("sin dato" if np.isnan(obs) else "con dato")})
    N = pd.DataFrame(rows)
    E = []
    for _, e in topo.iterrows():
        if e["linea_origen"] in POS and e["linea_destino"] in POS:
            (x1, y1), (x2, y2) = POS[e["linea_origen"]], POS[e["linea_destino"]]
            E.append({"x": x1, "y": y1, "x2": x2, "y2": y2, "estacion": e["estacion_transferencia"],
                      "par": f"{short(e['linea_origen'])}–{short(e['linea_destino'])}"})
    E = pd.DataFrame(E)
    if hour_fixed is None:
        hour = alt.param(name="hora_sel", value=17, bind=alt.binding_range(min=4, max=22, step=1, name="Hora (reproducción) "))
    else:
        hour = alt.param(name="hora_sel", value=int(hour_fixed))
    ax = alt.Axis(labels=False, ticks=False, grid=False, domain=False, title=None)
    edges = alt.Chart(E).mark_rule(color="#8a8a85", strokeWidth=2).encode(
        x=alt.X("x:Q", axis=ax, scale=alt.Scale(domain=[-5, 5])), y=alt.Y("y:Q", axis=ax, scale=alt.Scale(domain=[-2.2, 3.8])),
        x2="x2", y2="y2", tooltip=[alt.Tooltip("par:N", title="Conexión"), alt.Tooltip("estacion:N", title="Estación de integración")])
    base = alt.Chart(N).transform_filter("datum.hora == hora_sel")
    tips = [alt.Tooltip("linea:N", title="Línea"), alt.Tooltip("modo:N", title="Modo"), alt.Tooltip("hora:Q", title="Hora"),
            alt.Tooltip("observado:Q", title="Entradas registradas (real)", format=",.0f"),
            alt.Tooltip("esperado:Q", title="Esperado (estimación)", format=",.0f"),
            alt.Tooltip("z:Q", title="z", format=".1f"), alt.Tooltip("estado:N", title="Estado")]
    # Capa base gris para todos los nodos (incluidos los sin dato o sin modelar) y capa de color con z encima.
    under = base.mark_circle(size=1500, stroke="#1a1a19", strokeWidth=1.5, opacity=1, color="#c9c9c4").encode(
        x="x:Q", y="y:Q", tooltip=tips)
    pts = base.transform_filter("isValid(datum.zc)").mark_circle(size=1500, stroke="#1a1a19", strokeWidth=1.5, opacity=1).encode(
        x="x:Q", y="y:Q",
        color=alt.Color("zc:Q", title="z (desviación)", scale=alt.Scale(domain=[-Z_CLIP, Z_CLIP], range=DIV),
                        legend=alt.Legend(orient="bottom", direction="horizontal", gradientLength=260)),
        tooltip=tips)
    lab = base.mark_text(fontSize=15, fontWeight="bold", color="#111111").encode(x="x:Q", y="y:Q", text="etiqueta:N")
    title = "Relaciones entre líneas (esquema, no geográfico)" + ("" if hour_fixed is None else f" — {int(hour_fixed):02d}:00")
    return (edges + under + pts + lab).add_params(hour).properties(height=420, title=title)


def heatmap(h_day: pd.DataFrame, order):
    H = h_day.copy()
    H["linea_c"] = H["linea"].map(short)
    H["zc"] = H["z"].clip(-Z_CLIP, Z_CLIP)
    return alt.Chart(H).mark_rect(stroke="#fcfcfb", strokeWidth=2).encode(
        x=alt.X("hora:O", title="Hora"), y=alt.Y("linea_c:N", title="Línea", sort=[short(x) for x in order]),
        color=alt.Color("zc:Q", title="z", scale=alt.Scale(domain=[-Z_CLIP, Z_CLIP], range=DIV)),
        tooltip=[alt.Tooltip("linea:N", title="Línea"), alt.Tooltip("hora:O", title="Hora"),
                 alt.Tooltip("observado:Q", title="Real", format=",.0f"), alt.Tooltip("esperado:Q", title="Esperado", format=",.0f"),
                 alt.Tooltip("z:Q", title="z", format=".1f")]).properties(height=330)


def compare_chart(h_day: pd.DataFrame, ln: str):
    """Real frente a esperado para una línea (un gráfico por línea, ancho del contenedor: se adapta al móvil)."""
    H = h_day[h_day["linea"] == ln]
    L = H.melt(id_vars=["hora"], value_vars=["observado", "esperado"], var_name="serie", value_name="entradas")
    L["serie"] = L["serie"].map({"observado": "Real", "esperado": "Esperado"})
    dom = ["Real", "Esperado"]
    ch = alt.Chart(L).mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=36)).encode(
        x=alt.X("hora:O", title="Hora"), y=alt.Y("entradas:Q", title="Entradas por hora", axis=alt.Axis(labelExpr=Y_FMT)),
        color=alt.Color("serie:N", title=None, scale=alt.Scale(domain=dom, range=["#2a78d6", "#8a8a85"]), legend=alt.Legend(orient="top")),
        strokeDash=alt.StrokeDash("serie:N", legend=None, scale=alt.Scale(domain=dom, range=[[1, 0], [6, 4]])),
        tooltip=[alt.Tooltip("hora:O", title="Hora"), alt.Tooltip("serie:N", title="Serie"),
                 alt.Tooltip("entradas:Q", title="Entradas", format=",.0f")])
    return ch.properties(height=200, title=f"Línea {short(ln)}")


def forecast_chart(F: pd.DataFrame, kind: str):
    base = alt.Chart(F).encode(x=alt.X("hora:O", title="Hora"))
    band = base.mark_area(opacity=0.25, color="#2a78d6").encode(
        y=alt.Y("p10:Q", title="Entradas por hora", axis=alt.Axis(labelExpr=Y_FMT)), y2="p90:Q",
        tooltip=[alt.Tooltip("hora:O", title="Hora"), alt.Tooltip("p10:Q", title="p10", format=",.0f"), alt.Tooltip("p90:Q", title="p90", format=",.0f")])
    pred = base.mark_line(strokeWidth=2, strokeDash=[6, 4], color="#1f4e9a").encode(
        y=alt.Y("pred:Q", title="Entradas por hora", axis=alt.Axis(labelExpr=Y_FMT)), tooltip=[alt.Tooltip("hora:O", title="Hora"), alt.Tooltip("pred:Q", title="Predicción", format=",.0f")])
    layers = [band, pred] if F["p10"].notna().any() else [pred]
    if kind == "replay":
        obs = base.mark_line(strokeWidth=2, color="#111111", point=alt.OverlayMarkDef(size=30, color="#111111")).encode(
            y="observado:Q", tooltip=[alt.Tooltip("hora:O", title="Hora"), alt.Tooltip("observado:Q", title="Real", format=",.0f")])
        layers.append(obs)
    return alt.layer(*layers).properties(height=300)


def main():
    if not (ART / "manifiesto.json").exists():
        st.error("Faltan los artefactos de la plataforma. Ejecute `python scripts/fase3/build_plataforma.py`.")
        st.stop()
    man, horas, dias, fc, nxt, lineas, topo = load()
    lines = list(dict.fromkeys(dias["linea"]))
    st.title("Red multilínea — analítica operacional")
    st.caption("Granularidad: entradas registradas por **línea y hora** (export diario). No localiza estaciones, no identifica "
               "fallas técnicas y no anticipa incidentes. Cada bloque indica si muestra dato real, estimación, simulación o predicción.")
    with st.sidebar:
        st.subheader("Cómo leer esta página")
        for k, (lab, col) in KIND.items():
            st.markdown(f"<span style='background:{col};color:#fff;border-radius:6px;padding:1px 7px;font-weight:600'>{lab}</span> "
                        f"{man['tipos_de_dato'][k].split(': ', 1)[1]}", unsafe_allow_html=True)
        st.markdown("**z**: desviación frente a lo esperado para ese tipo de día y hora (rojo = menos entradas, azul = más). "
                    "|z| > 3 es inusual.")
        st.markdown(f"<small>Artefactos: commit {man.get('git_commit_build')} · datos SHA-256 {man['sha256_datos'][:12]} · "
                    f"creado {man['creado']}</small>", unsafe_allow_html=True)

    dmin, dmax = dias["fecha"].min().date(), dias["fecha"].max().date()
    c1, c2 = st.columns([1, 3])
    with c1:
        d = st.date_input("Día a revisar", value=pd.Timestamp("2023-08-28").date(), min_value=dmin, max_value=dmax, format="DD/MM/YYYY")
    t = pd.Timestamp(d)
    h_day = horas[horas["fecha"] == t]
    d_day = dias[dias["fecha"] == t].set_index("linea").reindex(lines)
    with c2:
        n_anom = d_day["categoria"].astype(str).str.startswith("anomalia").sum()
        note = d_day["nota_calendario"].dropna().astype(str)
        note = next((x for x in note if x), "")
        msg = f"**{t.strftime('%d/%m/%Y')}** · {n_anom} línea(s) con anomalía estadística"
        if note:
            msg += f" · Calendario: {note}"
        (st.warning if n_anom else st.info)(msg)

    tab1, tab2, tab3, tab4 = st.tabs(["🕸️ Red y día", "📈 Pronóstico", "🧪 Simulación", "📚 Modelos y datos"])
    with tab1:
        a, b = st.columns([1.15, 1])
        with a:
            badge("estimacion", "Color = z de cada línea en la hora elegida con el control deslizante (reproducción hora a hora).")
            ph = st.empty()
            if st.button("▶ Reproducir el día hora a hora (04–22 h)", key="play"):
                for h in range(4, 23):  # animación: un fotograma por hora
                    ph.altair_chart(network_chart(h_day, lineas, topo, set(lines), hour_fixed=h), use_container_width=True)
                    time.sleep(0.6)
            ph.altair_chart(network_chart(h_day, lineas, topo, set(lines)), use_container_width=True)
            st.caption("Aristas = estaciones de integración verificadas en las páginas oficiales del Metro (archivo data/topologia_red.csv). "
                       "La Línea L (gris) no se modela: no opera todas las horas.")
        with b:
            badge("estimacion", "Desviación por línea y hora (z) del día")
            st.altair_chart(heatmap(h_day, lines), use_container_width=True)
        st.subheader("Evaluación del día por línea")
        badge("estimacion", "Motor de reglas: medición > ausencia simultánea > datos > anomalía > normal")
        cats = man["categorias"]
        tbl = pd.DataFrame({
            "Línea": [short(x) for x in d_day.index],
            "Resultado": [f"{CAT_ICON.get(c, '')} {cats.get(c, c)}" for c in d_day["categoria"].astype(str)],
            "Score / umbral": [f"{num(s, 1)} / {num(u, 1)}" for s, u in zip(d_day["score"], d_day["umbral"])],
            "Detalle": [detalle(h, a, o) for h, a, o in zip(d_day["horas_fuera"].astype(str), d_day["lineas_acompanan"].astype(str),
                                                           d_day["lineas_sentido_opuesto"].astype(str))],
        })
        st.table(tbl.set_index("Línea"))
        st.caption("El alcance (local / varias líneas / generalizada) es **orientativo**: su exactitud con anomalías simuladas fue 0,884, "
                   "por debajo del 0,90 pre-registrado. 'Sentido opuesto' es descriptivo (posible redistribución); no prueba causalidad.")
        st.subheader("Real frente a esperado")
        sel = st.multiselect("Líneas", lines, default=[x for x in ["LINEA A", "LINEA B", "LINEA 2"] if x in lines], format_func=short)
        if sel:
            badge("real", "línea continua · ")
            badge("estimacion", "línea discontinua")
            cols = st.columns(min(3, len(sel)))
            for i, ln_c in enumerate(sel):
                with cols[i % len(cols)]:
                    st.altair_chart(compare_chart(h_day, ln_c), use_container_width=True)

    with tab2:
        st.subheader("Pronóstico del día siguiente por línea y hora")
        ln = st.selectbox("Línea", lines, format_func=short, key="fc_line")
        F = fc[(fc["linea"] == ln) & (fc["fecha"] == t)].sort_values("hora")
        if len(F):
            badge("prediccion", f"línea discontinua y banda del 80 %, emitida al cierre del {(t - pd.Timedelta(days=1)).strftime('%d/%m/%Y')} sin conocer el día (reproducción fuera de muestra)")
            badge("real", "línea negra continua")
            st.altair_chart(forecast_chart(F, "replay"), use_container_width=True)
            w = float(np.abs(F["observado"] - F["pred"]).sum() / F["observado"].sum())
            m1, m2, m3 = st.columns(3)
            m1.metric("Error del día (WAPE)", f"{num(100 * w, 1)} %")
            m2.metric("Modelo", F["modelo"].iloc[0])
            cov = ((F["observado"] >= F["p10"]) & (F["observado"] <= F["p90"])).mean() if F["p10"].notna().any() else np.nan
            m3.metric("Horas dentro del intervalo 80 %", "—" if np.isnan(cov) else f"{num(100 * cov, 0)} %")
        else:
            st.info("Hay pronósticos fuera de muestra para 01/07/2023–30/06/2024 (periodo de evaluación). Elija un día de ese periodo.")
        N = nxt[nxt["linea"] == ln].sort_values("hora")
        if len(N):
            from metro_anomaly.calendar import event_note
            nd = N["fecha"].iloc[0]
            st.subheader(f"Próximo día sin datos: {nd.strftime('%d/%m/%Y')}")
            if event_note(nd):
                st.caption(f"Calendario: {event_note(nd)}")
            badge("prediccion", "línea discontinua y banda del 80 %; aún no hay dato real para compararla")
            st.altair_chart(forecast_chart(N, "next"), use_container_width=True)
            r = float(N["ratio_p95_gbm"].iloc[0])
            if ln in man.get("lineas_h6_validadas", []):
                nivel = "alto" if r > 1 else "normal"
                st.metric("Demanda total pronosticada frente al p95 del último año", f"{num(100 * r, 0)} %",
                          help="Por encima de 100 %: posible día de alta demanda (AUC validado > 0,80 en esta línea).")
                st.caption(f"Riesgo de alta demanda: **{nivel}** · {'⬆️' if r > 1 else '➖'}")
            else:
                st.caption("El indicador de día de alta demanda no se muestra para esta línea: no superó la validación (H6).")
        st.caption("Banda = intervalo del 80 % con calibración conformal en línea. Evidencia y limitaciones: reports/fase3/pronostico_holdout.md.")

    with tab3:
        st.subheader("¿Cómo se clasificaría una caída hipotética?")
        badge("simulacion", "escenario sintético aplicado al día elegido; no ocurrió")
        s1, s2, s3 = st.columns(3)
        sl = s1.selectbox("Línea afectada", lines, format_func=short, key="sim_line")
        hrs = s2.slider("Horas", 4, 22, (17, 19))
        f = s3.slider("Factor sobre las entradas", 0.1, 1.5, 0.5, 0.05)
        afect_red = st.checkbox("Aplicar el mismo factor a todas las líneas (evento de red)")
        H = h_day.copy()
        if H.empty:
            st.info("No hay datos para el día elegido.")
        else:
            mask = H["hora"].between(*hrs) & ((H["linea"] == sl) | afect_red)
            # z simulado = (log1p(f · observado) − log esperado) / sigma, con la escala del perfil de cada línea y hora
            H.loc[mask, "observado"] = H.loc[mask, "observado"] * f
            H.loc[mask, "z"] = (np.log1p(H.loc[mask, "observado"]) - np.log1p(H.loc[mask, "esperado"])) / H.loc[mask, "sigma"]
            from metro_anomaly.red import classify_scope
            zt = H[H["linea"] == sl].set_index("hora")["z"]
            others = {l2: g.set_index("hora")["z"] for l2, g in H.groupby("linea") if l2 != sl}
            r = classify_scope(zt.dropna(), others)
            st.altair_chart(heatmap(H, lines), use_container_width=True)
            st.markdown(f"Alcance estimado para la Línea {short(sl)}: **{r['alcance']}** "
                        f"(fracción de otras líneas que acompañan: {num(r['fraccion_lineas'], 2)}; {r['sentido']}).")
            st.caption("Recalcula z con el perfil de cada línea; ilustra el clasificador de alcance (exactitud validada 0,884), no es una evaluación.")

    with tab4:
        st.subheader("Catálogo de capacidades y estado de la evidencia")
        st.table(pd.DataFrame(man["catalogo_modelos"]).rename(columns={"id": "Componente", "capacidad": "Capacidad", "estado": "Estado",
                                                                        "evidencia": "Evidencia", "nota": "Resultado"}).set_index("Componente"))
        st.subheader("Umbrales por línea (p95 de DEV)")
        st.table(pd.DataFrame(man["umbrales"]).T.rename(columns={"score": "Score", "umbral": "Umbral", "n_dev": "Días DEV"}))
        st.subheader("Lo que esta plataforma NO puede hacer con los datos actuales")
        st.markdown("- Localizar la estación o el tramo afectado (se necesitan datos por estación y 15 min).\n"
                    "- Identificar fallas técnicas o priorizar mantenimiento (se necesitan telemetría y registros de mantenimiento).\n"
                    "- Anticipar incidentes específicos con horas de antelación (se necesita telemetría en tiempo real).\n"
                    "- Ver interrupciones parciales: los incidentes del 05/02/2023 y 22/08/2023 no dejan señal en línea × hora.\n"
                    "Hoja de ruta de datos: `docs/fase3/HOJA_DE_RUTA_DATOS.md`.")


main()
