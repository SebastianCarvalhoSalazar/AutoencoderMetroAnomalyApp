// Genera capturas anotadas de la app (guía visual). Uso:
// node tests/e2e/guia_visual.js <url> <dir_fixtures> <dir_salida>
const { chromium } = require('playwright');
const path = require('path');
const [url, fx, out] = process.argv.slice(2);

async function idle(p) {
  await p.waitForTimeout(300);
  for (let i = 0; i < 120 && await p.locator('[data-testid="stStatusWidget"]').count(); i++) await p.waitForTimeout(250);
  await p.waitForTimeout(1500);
}

async function setDate(p, ddmmyyyy) {
  const inp = p.locator('[data-testid="stDateInput"] input').first();
  await inp.click(); await inp.press('Control+A'); await inp.press('Backspace');
  for (let i = 0; i < 12; i++) await inp.press('Backspace');
  await inp.pressSequentially(ddmmyyyy, { delay: 25 }); await inp.press('Enter');
  await idle(p); await p.keyboard.press('Escape'); await idle(p);
  if ((await inp.inputValue()) !== ddmmyyyy) throw new Error('fecha no aplicada ' + ddmmyyyy);
}

// Dibuja un recuadro y una etiqueta numerada sobre la unión de los elementos que coinciden con `sel`.
async function annotate(p, notes) {
  await p.evaluate((notes) => {
    document.querySelectorAll('.guia-anot').forEach(e => e.remove());
    for (const n of notes) {
      let els = Array.from(document.querySelectorAll(n.sel));
      if (n.text) els = els.filter(e => (e.innerText || '').includes(n.text));
      if (n.idx !== undefined) els = els.slice(n.idx, n.idx + 1);
      els = els.filter(e => e.getBoundingClientRect().width > 0);
      if (!els.length) continue;
      const rs = els.map(e => e.getBoundingClientRect());
      const x = Math.min(...rs.map(r => r.left)) - 6, y = Math.min(...rs.map(r => r.top)) - 6;
      const w = Math.max(...rs.map(r => r.right)) - x + 6, h = Math.max(...rs.map(r => r.bottom)) - y + 6;
      const box = document.createElement('div');
      box.className = 'guia-anot';
      Object.assign(box.style, { position: 'fixed', left: x + 'px', top: y + 'px', width: w + 'px', height: h + 'px',
        border: '3px solid #d9480f', borderRadius: '8px', zIndex: 2147483646, pointerEvents: 'none', boxSizing: 'border-box' });
      const lab = document.createElement('div');
      lab.className = 'guia-anot';
      lab.textContent = n.label;
      // inside: etiqueta dentro del recuadro (esquina superior derecha); right: alineada a la derecha por encima
      const ly = n.insideBottom ? y + h - 34 : (n.inside ? y + 4 : (n.below ? y + h + 4 : Math.max(y - 30, 2)));
      const lx = n.right ? Math.max(x + w - (n.width || 570), 2) : Math.max(x, 2);
      Object.assign(lab.style, { position: 'fixed', left: lx + 'px', top: ly + 'px', background: '#d9480f', color: '#fff',
        font: '600 14px/1.25 system-ui, sans-serif', padding: '4px 9px', borderRadius: '6px', zIndex: 2147483647, maxWidth: '560px',
        boxShadow: '0 1px 4px rgba(0,0,0,.3)', pointerEvents: 'none' });
      document.body.appendChild(box); document.body.appendChild(lab);
    }
  }, notes);
  await p.waitForTimeout(300);
}

(async () => {
  const b = await chromium.launch();
  const ctx = await b.newContext({ viewport: { width: 1440, height: 2300 }, deviceScaleFactor: 1 });
  const p = await ctx.newPage();
  await p.goto(url, { waitUntil: 'networkidle' });
  await p.waitForSelector('[data-testid="stMain"] [data-testid="stAlert"]', { timeout: 120000 });
  await idle(p);

  // 1. Vista general: día con anomalía (incidente real del 28/08/2023)
  await setDate(p, '28/08/2023');
  await annotate(p, [
    { sel: '[data-testid="stDateInput"]', label: '1 · Elige el día a revisar (dd/mm/aaaa)', right: true },
    { sel: '[data-testid="stMain"] [data-testid="stAlert"]', idx: 0, label: '2 · Veredicto: anomalía estadística + severidad; aclara que NO es un incidente confirmado', right: true },
    { sel: '[data-testid="stMetric"]', label: '3 · Score frente al umbral, frecuencia histórica y desviación del día', below: true },
    { sel: '[data-testid="stSidebar"] li', text: 'De 3 incidentes', label: '4 · Límites del sistema, siempre visibles', below: true },
  ]);
  await p.screenshot({ path: path.join(out, '01_vista_general_anomalia.png'), clip: { x: 0, y: 0, width: 1440, height: 900 } });

  // 2. Gráfica horaria y evolución temporal
  await annotate(p, [
    { sel: '[data-testid="stVegaLiteChart"], .vega-embed', idx: 0, label: '5 · Entradas por hora: observado (continua) frente a esperado (discontinua) y rango normal (banda). Rombos rojos = horas fuera de rango (18 y 19 h)', right: true, inside: true },
    { sel: '[data-testid="stVegaLiteChart"], .vega-embed', idx: 1, label: '6 · Evolución del score de los últimos 60 días: el pico rojo supera el umbral' },
    { sel: '[data-testid="stExpander"]', label: '7 · Tabla por hora (valores exactos, z y horas fuera de rango)', below: true },
  ]);
  const top = await p.locator('[data-testid="stVegaLiteChart"], .vega-embed').first().boundingBox();
  await p.screenshot({ path: path.join(out, '02_graficas_hora_y_tendencia.png'), clip: { x: 300, y: Math.max(top.y - 80, 0), width: 1140, height: 1050 } });

  // 3. Día normal
  await setDate(p, '06/03/2024');
  await annotate(p, [
    { sel: '[data-testid="stMain"] [data-testid="stAlert"]', idx: 0, label: 'Sin anomalía (no descarta interrupciones parciales)', right: true, width: 410 },
    { sel: '[data-testid="stVegaLiteChart"], .vega-embed', idx: 0, label: 'Lo observado sigue el perfil esperado y queda dentro de la banda' },
  ]);
  await p.screenshot({ path: path.join(out, '03_dia_normal.png'), clip: { x: 300, y: 380, width: 1140, height: 1050 } });

  // 4. Efecto de medición: elecciones
  await setDate(p, '29/10/2023');
  await annotate(p, [
    { sel: '[data-testid="stMain"] [data-testid="stAlert"]', idx: 0, label: 'Efecto de medición (no cobro electoral)', right: true, width: 330 },
    { sel: '[data-testid="stMain"] [data-testid="stAlert"]', idx: 1, label: 'Calendario automático', right: true, inside: true, width: 180 },
  ]);
  await p.screenshot({ path: path.join(out, '04_efecto_de_medicion_elecciones.png'), clip: { x: 300, y: 380, width: 1140, height: 820 } });

  // 5. Carga de archivo con anomalía inyectada
  await p.getByRole('tab', { name: /Cargar archivo/ }).click(); await idle(p);
  await p.setInputFiles('input[type=file]', path.join(fx, '03_anomalia_inyectada_2024-06-12.xlsx')); await idle(p);
  await annotate(p, [
    { sel: '[data-testid="stFileUploader"]', label: '1 · Sube el archivo oficial (.xlsx/.csv) con uno o varios días', right: true, inside: true, width: 480 },
    { sel: '[data-testid="stMain"] [data-testid="stAlert"]', text: 'evaluados', label: '2 · Resumen del archivo', right: true, width: 200 },
    { sel: '[data-testid="stDataFrame"]', label: '3 · Tabla de resultados por día (ordenable)', right: true, insideBottom: true, width: 360 },
    { sel: '[data-testid="stDownloadButton"]', label: '4 · Exportar resultados (CSV)', right: true, inside: true, width: 280 },
    { sel: '[data-testid="stSelectbox"]', label: '5 · Ver el detalle de un día (las anomalías aparecen primero)', right: true, inside: true, width: 500 },
  ]);
  const up = await p.locator('[data-testid="stFileUploader"]').boundingBox();
  await p.screenshot({ path: path.join(out, '05_carga_de_archivo.png'), clip: { x: 300, y: Math.max(up.y - 120, 0), width: 1140, height: 1150 } });

  // 6. Detalle del día inyectado
  const al = p.locator('[data-testid="stMain"] [data-testid="stAlert"]', { hasText: 'Anomalía estadística detectada' }).last();
  await al.scrollIntoViewIfNeeded();
  await annotate(p, [
    { sel: '[data-testid="stMain"] [data-testid="stAlert"]', text: 'Anomalía estadística detectada', label: 'Caída simulada 17–18 h: detectada y localizada', right: true, width: 380 },
  ]);
  const ab = await al.boundingBox();
  await p.screenshot({ path: path.join(out, '06_detalle_dia_cargado.png'), clip: { x: 300, y: Math.max(ab.y - 120, 0), width: 1140, height: 900 } });

  // 7. Problema de datos (día en ceros)
  await p.setInputFiles('input[type=file]', path.join(fx, '13_dia_en_ceros.xlsx')); await idle(p);
  const dq = p.locator('[data-testid="stMain"] [data-testid="stAlert"]', { hasText: 'Posible problema de datos' }).last();
  await dq.scrollIntoViewIfNeeded();
  await annotate(p, [
    { sel: '[data-testid="stMain"] [data-testid="stAlert"]', text: 'Posible problema de datos', label: 'Control de calidad: día en ceros = posible falla de captura', right: true, width: 470 },
  ]);
  const db = await dq.boundingBox();
  await p.screenshot({ path: path.join(out, '07_problema_de_datos.png'), clip: { x: 300, y: Math.max(db.y - 120, 0), width: 1140, height: 700 } });
  await b.close();
  console.log('OK');
})().catch(e => { console.error('FALLO', e); process.exit(1); });
