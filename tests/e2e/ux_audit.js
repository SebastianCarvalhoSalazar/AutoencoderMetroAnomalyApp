// Auditoría funcional, visual y de UX de la app (Playwright). Registra observaciones; las aserciones se hacen
// en tests/e2e/check_ux_audit.py comparando contra el backend.
// Uso: node tests/e2e/ux_audit.js <url> <dir_fixtures> <dir_capturas> <salida.json> [axe.min.js]
const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');
const [url, fx, shots, outJson, axePath] = process.argv.slice(2);
fs.mkdirSync(shots, { recursive: true });

const results = { steps: [], axe: {}, responsive: {}, keyboard: {} };
let consoleErrors = [], pageErrors = [], failedRequests = [];

async function idle(p) {
  await p.waitForTimeout(250);
  for (let i = 0; i < 120; i++) {
    const running = await p.locator('[data-testid="stStatusWidget"]').count();
    if (!running) break;
    await p.waitForTimeout(250);
  }
  await p.waitForTimeout(400);
}

async function snapshot(p, id, extra = {}) {
  await idle(p);
  const file = path.join(shots, `${id}.png`);
  await p.screenshot({ path: file, fullPage: true });
  const texts = async (sel) => (await p.locator(sel).allInnerTexts()).map(t => t.trim()).filter(Boolean);
  const step = {
    id, ...extra, screenshot: file,
    alerts: await texts('[data-testid="stAlert"]'),
    markdown: await texts('[data-testid="stMarkdownContainer"]'),
    exceptions: await p.locator('[data-testid="stException"]').count(),
    charts: await p.locator('[data-testid="stVegaLiteChart"], .vega-embed').count(),
    dataframes: await p.locator('[data-testid="stDataFrame"]').count(),
    tables: await p.locator('[data-testid="stTable"] table').count(),
    tableText: await texts('[data-testid="stTable"] table'),
    downloads: await p.locator('[data-testid="stDownloadButton"]').count(),
    metricLabels: await texts('[data-testid="stMetricLabel"]'),
    metrics: await texts('[data-testid="stMetricValue"]'),
    subheaders: await texts('[data-testid="stHeadingWithActionElements"] h3, h3'),
    consoleErrors: consoleErrors.splice(0), pageErrors: pageErrors.splice(0), failedRequests: failedRequests.splice(0),
  };
  results.steps.push(step);
  console.log(`[${id}] alerts=${JSON.stringify(step.alerts).slice(0, 220)} exc=${step.exceptions} js=${step.consoleErrors.length + step.pageErrors.length} req=${step.failedRequests.length}`);
  return step;
}

// Nota: el campo de fecha de Streamlit (BaseWeb, con máscara) ignora fill(); hay que escribir carácter a carácter
// y verificar que el valor quedó aplicado (un E2E anterior "pasaba" sin cambiar realmente la fecha).
async function setDate(p, iso) {
  const inp = p.locator('[data-testid="stDateInput"] input').first();
  const cur = await inp.inputValue();
  const [y, m, d] = iso.split('-');
  const val = /^\d{4}\//.test(cur) ? `${y}/${m}/${d}` : `${d}/${m}/${y}`;
  // La máscara puede conservar dígitos del valor previo: se vacía por completo y se reintenta hasta 3 veces.
  for (let attempt = 0; attempt < 3; attempt++) {
    await inp.click();
    await inp.press('Control+A'); await inp.press('Backspace');
    for (let i = 0; i < 12; i++) await inp.press('Backspace');
    await inp.pressSequentially(val, { delay: 25 }); await inp.press('Enter');
    await idle(p);
    await p.keyboard.press('Escape');
    await idle(p);
    if ((await inp.inputValue()) === val) break;
  }
  return { typed: val, applied: (await inp.inputValue()) === val, valueAfter: await inp.inputValue() };
}

async function gotoTab(p, name) {
  await p.getByRole('tab', { name }).click();
  await idle(p);
}

async function upload(p, file) {
  await p.setInputFiles('input[type=file]', path.join(fx, file));
  await idle(p);
}

async function runAxe(p, id) {
  if (!axePath) return;
  await p.addScriptTag({ path: axePath });
  const r = await p.evaluate(async () => {
    const res = await window.axe.run(document, { runOnly: { type: 'tag', values: ['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa'] } });
    return res.violations.map(v => ({ id: v.id, impact: v.impact, help: v.help, nodes: v.nodes.length,
      targets: v.nodes.slice(0, 4).map(n => n.target.join(' ')) }));
  });
  results.axe[id] = r;
  console.log(`[axe ${id}] ${r.map(v => `${v.id}(${v.impact},${v.nodes})`).join(' ') || 'sin violaciones'}`);
}

async function overflow(p) {
  return p.evaluate(() => {
    const main = document.querySelector('[data-testid="stMain"], section.main, [data-testid="stAppViewContainer"]');
    const doc = document.documentElement;
    return { docScroll: doc.scrollWidth, docClient: doc.clientWidth,
      mainScroll: main ? main.scrollWidth : null, mainClient: main ? main.clientWidth : null };
  });
}

(async () => {
  const b = await chromium.launch();
  const ctx = await b.newContext({ viewport: { width: 1440, height: 1100 }, acceptDownloads: true });
  const p = await ctx.newPage();
  p.on('console', m => { if (m.type() === 'error') consoleErrors.push(m.text()); });
  p.on('pageerror', e => pageErrors.push(String(e)));
  p.on('requestfailed', r => failedRequests.push(`${r.url()} ${r.failure() && r.failure().errorText}`));
  p.on('response', r => { if (r.status() >= 400) failedRequests.push(`${r.status()} ${r.url()}`); });

  const t0 = Date.now();
  await p.goto(url, { waitUntil: 'networkidle' });
  await p.waitForSelector('[data-testid="stMain"] [data-testid="stAlert"]', { timeout: 120000 });
  await snapshot(p, 'H01_carga_inicial', { loadMs: Date.now() - t0 });
  await runAxe(p, 'H01_carga_inicial');

  const days = [['H02_dia_normal', '2024-03-06'], ['H03_anomalia_2023-08-28', '2023-08-28'], ['H04_navidad', '2023-12-25'],
    ['H05_elecciones', '2023-10-29'], ['H06_hora_faltante_2023-12-24', '2023-12-24'], ['H07_horas_faltantes_2019-02-16', '2019-02-16'],
    ['H08_sin_datos_2020-06-13', '2020-06-13'], ['H09_fecha_minima', '2019-03-01'], ['H10_fecha_maxima', '2024-06-30'],
    ['H11_semana_santa_2024-03-27', '2024-03-27'], ['H12_aumento_2023-12-14', '2023-12-14']];
  for (const [id, d] of days) { const r = await setDate(p, d); await snapshot(p, id, { date: d, dateInput: r }); }

  // Fecha fuera de rango escrita a mano
  const rOut = await setDate(p, '2018-01-01');
  await snapshot(p, 'H13_fecha_fuera_de_rango', { dateInput: rOut });

  // Expansor de detalle
  await setDate(p, '2023-08-28');
  const exp = p.locator('[data-testid="stExpander"] summary').first();
  if (await exp.count()) { await exp.click(); await idle(p); }
  await snapshot(p, 'H14_detalle_por_hora_abierto', { date: '2023-08-28' });
  await runAxe(p, 'H14_detalle_por_hora');

  // Línea de tiempo (si existe)
  results.timeline = await p.locator('text=/Evolución del score/i').count();

  // Teclado: recorrido con Tab
  await p.mouse.click(700, 5);
  const focused = [];
  for (let i = 0; i < 25; i++) {
    await p.keyboard.press('Tab');
    focused.push(await p.evaluate(() => { const a = document.activeElement; return a ? `${a.tagName}${a.getAttribute('role') ? '[' + a.getAttribute('role') + ']' : ''}:${(a.innerText || a.getAttribute('aria-label') || '').slice(0, 30)}` : null; }));
  }
  results.keyboard.tabOrder = focused;

  // Pestaña de carga
  await gotoTab(p, /Cargar archivo/);
  await snapshot(p, 'U00_pestana_carga_vacia');
  const files = fs.readdirSync(fx).sort();
  for (const f of files) {
    await upload(p, f);
    const st = await snapshot(p, `U_${f.replace(/\.[a-z]+$/, '')}`, { file: f });
    if (f.startsWith('03_') || f.startsWith('05_') || f.startsWith('16_')) {
      await runAxe(p, `U_${f}`);
      // descarga (si existe)
      const dl = p.locator('[data-testid="stDownloadButton"] button');
      if (await dl.count()) {
        const [d] = await Promise.all([p.waitForEvent('download'), dl.first().click()]);
        const target = path.join(shots, `descarga_${f}.csv`);
        await d.saveAs(target); st.download = target;
      }
      // selector de día: elegir otro día y verificar que cambia el detalle
      const sb = p.locator('[data-testid="stSelectbox"]');
      if (await sb.count()) {
        await sb.first().click(); await p.waitForTimeout(300);
        const opts = p.locator('[role="option"]');
        const n = await opts.count();
        if (n > 1) { await opts.nth(1).click(); await idle(p); }
        await snapshot(p, `U_${f.replace(/\.[a-z]+$/, '')}_otro_dia`, { file: f, selectedOption: 1 });
      }
    }
  }

  // Responsivo (390 px)
  const m = await b.newContext({ viewport: { width: 390, height: 844 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true });
  const mp = await m.newPage();
  mp.on('pageerror', e => pageErrors.push(String(e)));
  await mp.goto(url, { waitUntil: 'networkidle' });
  await mp.waitForSelector('[data-testid="stAlert"]', { timeout: 120000 });
  await idle(mp);
  results.responsive.H03 = await overflow(mp);
  await mp.screenshot({ path: path.join(shots, 'M01_movil_historico.png'), fullPage: true });
  await mp.getByRole('tab', { name: /Cargar archivo/ }).click(); await idle(mp);
  await mp.setInputFiles('input[type=file]', path.join(fx, '03_anomalia_inyectada_2024-06-12.xlsx')); await idle(mp);
  results.responsive.U03 = await overflow(mp);
  await mp.screenshot({ path: path.join(shots, 'M02_movil_carga.png'), fullPage: true });
  console.log('[responsive]', JSON.stringify(results.responsive));

  fs.writeFileSync(outJson, JSON.stringify(results, null, 1));
  await b.close();
  console.log('AUDITORIA COMPLETA');
})().catch(e => { console.error('FALLO', e); fs.writeFileSync(outJson, JSON.stringify(results, null, 1)); process.exit(1); });
