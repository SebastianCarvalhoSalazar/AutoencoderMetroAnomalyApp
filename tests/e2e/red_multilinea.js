// E2E de la página multilínea: capturas (escritorio y móvil), extracción de la tabla para la verificación
// visual-datos, reproducción por hora, pestañas, accesibilidad (axe) y tiempos.
// Uso: node tests/e2e/red_multilinea.js <url_base> <dir_salida> [ruta_axe.min.js]
const { chromium, devices } = require('playwright');
const fs = require('fs');
const path = require('path');
const [base, out, axePath] = process.argv.slice(2);
const url = base.replace(/\/$/, '') + '/Red_multilinea';

async function idle(p) {
  await p.waitForTimeout(300);
  for (let i = 0; i < 160 && await p.locator('[data-testid="stStatusWidget"]').count(); i++) await p.waitForTimeout(250);
  await p.waitForTimeout(1200);
}

async function setDate(p, ddmmyyyy) {
  const inp = p.locator('[data-testid="stDateInput"] input').first();
  await inp.click(); await inp.press('Control+A'); await inp.press('Backspace');
  for (let i = 0; i < 12; i++) await inp.press('Backspace');
  await inp.pressSequentially(ddmmyyyy, { delay: 25 }); await inp.press('Enter');
  await idle(p); await p.keyboard.press('Escape'); await idle(p);
  const v = await inp.inputValue();
  if (v !== ddmmyyyy) throw new Error('fecha no aplicada ' + ddmmyyyy + ' -> ' + v);
}

async function tableRows(p) {
  return p.evaluate(() => {
    const t = document.querySelector('[data-testid="stTable"] table');
    if (!t) return null;
    const head = Array.from(t.querySelectorAll('thead th')).map(x => x.innerText.trim());
    const rows = Array.from(t.querySelectorAll('tbody tr')).map(tr => Array.from(tr.querySelectorAll('th,td')).map(x => x.innerText.trim()));
    return { head, rows };
  });
}

async function checkPage(p, tag, res) {
  const err = await p.locator('[data-testid="stException"]').count();
  const overflow = await p.evaluate(() => {
    if (document.documentElement.scrollWidth > window.innerWidth + 1) return 'pagina';
    // contenido recortado a la derecha dentro de tablas o gráficos
    for (const el of document.querySelectorAll('[data-testid="stTable"], .vega-embed')) {
      const r = el.getBoundingClientRect();
      if (r.width > 0 && (r.right > window.innerWidth + 1 || el.scrollWidth > el.clientWidth + 1)) return el.className || 'stTable';
    }
    return false;
  });
  res.checks.push({ tag, excepciones: err, desbordamiento_horizontal: overflow });
}

(async () => {
  fs.mkdirSync(out, { recursive: true });
  const res = { url, checks: [], dias: {}, tiempos_ms: {}, axe: null };
  const b = await chromium.launch();
  const ctx = await b.newContext({ viewport: { width: 1440, height: 1800 }, deviceScaleFactor: 1 });
  const p = await ctx.newPage();
  const errors = [];
  p.on('pageerror', e => errors.push(String(e)));
  let t0 = Date.now();
  await p.goto(url, { waitUntil: 'networkidle' });
  await p.waitForSelector('[data-testid="stTable"]', { timeout: 180000 });
  await idle(p);
  res.tiempos_ms.carga_inicial = Date.now() - t0;
  await checkPage(p, 'inicial', res);
  for (const [d, name] of [['28/08/2023', 'in3'], ['29/10/2023', 'elecciones'], ['06/03/2024', 'normal'], ['24/12/2023', 'nochebuena']]) {
    t0 = Date.now();
    await setDate(p, d);
    res.tiempos_ms['cambio_fecha_' + name] = Date.now() - t0;
    res.dias[d] = await tableRows(p);
    await checkPage(p, name, res);
    await p.screenshot({ path: path.join(out, `R_${name}.png`), fullPage: true });
  }
  // Reproducción hora a hora: el control deslizante de Vega cambia el color de los nodos sin recargar.
  await setDate(p, '28/08/2023');
  const slider = p.locator('.vega-bindings input[type="range"]').first();
  const colors = {};
  for (const h of [8, 17, 19]) {
    await slider.evaluate((el, v) => { el.value = v; el.dispatchEvent(new Event('input', { bubbles: true })); }, String(h));
    await p.waitForTimeout(500);
    colors[h] = await p.evaluate(() => Array.from(document.querySelectorAll('.vega-embed'))[0]
      .querySelectorAll('path[fill], circle, .mark-symbol path').length);
    const box = await p.locator('.vega-embed').first().boundingBox();
    await p.screenshot({ path: path.join(out, `R_red_hora_${h}.png`), clip: { x: box.x, y: box.y, width: box.width, height: box.height + 40 } });
  }
  res.reproduccion_marcas = colors;
  // Pestañas
  for (const [tab, name] of [[/Pronóstico/, 'pronostico'], [/Simulación/, 'simulacion'], [/Modelos y datos/, 'catalogo']]) {
    t0 = Date.now();
    await p.getByRole('tab', { name: tab }).click(); await idle(p);
    res.tiempos_ms['pestana_' + name] = Date.now() - t0;
    await checkPage(p, name, res);
    await p.screenshot({ path: path.join(out, `R_${name}.png`), fullPage: true });
  }
  if (axePath && fs.existsSync(axePath)) {
    await p.getByRole('tab', { name: /Red y día/ }).click(); await idle(p);
    await p.addScriptTag({ path: axePath });
    res.axe = await p.evaluate(async () => {
      const r = await axe.run(document, { runOnly: ['wcag2a', 'wcag2aa'] });
      return r.violations.map(v => ({ id: v.id, impact: v.impact, n: v.nodes.length, ejemplo: v.nodes[0] && v.nodes[0].target.join(' ') }));
    });
  }
  // Móvil
  const m = await b.newContext({ ...devices['iPhone 13'] });
  const pm = await m.newPage();
  await pm.goto(url, { waitUntil: 'networkidle' });
  await pm.waitForSelector('[data-testid="stTable"]', { timeout: 180000 });
  await idle(pm);
  await checkPage(pm, 'movil', res);
  await pm.screenshot({ path: path.join(out, 'R_movil.png'), fullPage: true });
  // Streamlit desplaza un contenedor interno: se capturan los elementos clave por separado.
  for (const [sel, name] of [['.vega-embed', 'R_movil_red.png'], ['[data-testid="stTable"]', 'R_movil_tabla.png']]) {
    const el = pm.locator(sel).first();
    await el.scrollIntoViewIfNeeded(); await pm.waitForTimeout(400);
    await el.screenshot({ path: path.join(out, name) });
  }
  res.errores_js = errors;
  fs.writeFileSync(path.join(out, 'red_multilinea.json'), JSON.stringify(res, null, 1));
  await b.close();
  console.log('OK');
})().catch(e => { console.error('FALLO', e); process.exit(1); });
