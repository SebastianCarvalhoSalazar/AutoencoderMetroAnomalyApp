// E2E de la página de investigación: excepciones, desbordamiento, recorrido de los ejemplos y capturas.
// Uso: node tests/e2e/investigacion.js <url_base> <dir_salida>
const { chromium, devices } = require('playwright');
const fs = require('fs'); const path = require('path');
const [base, out] = process.argv.slice(2);
const url = base.replace(/\/$/, '') + '/Investigacion_modelos';
async function idle(p) { await p.waitForTimeout(300); for (let i = 0; i < 160 && await p.locator('[data-testid="stStatusWidget"]').count(); i++) await p.waitForTimeout(250); await p.waitForTimeout(900); }
async function check(p) {
  return p.evaluate(() => {
    const vw = document.documentElement.clientWidth; const bad = [];
    for (const el of document.querySelectorAll('[data-testid="stTable"], .vega-embed')) { const r = el.getBoundingClientRect(); if (r.width > 0 && r.right > vw + 1) bad.push(el.className || 'stTable'); }
    return { exceptions: document.querySelectorAll('[data-testid="stException"]').length, pageOverflow: document.documentElement.scrollWidth > vw + 1, clipped: bad.slice(0, 5) };
  });
}
(async () => {
  fs.mkdirSync(out, { recursive: true });
  const b = await chromium.launch(); const res = { checks: [] }; const errs = [];
  const p = await b.newPage({ viewport: { width: 1400, height: 4600 } });
  p.on('pageerror', (e) => errs.push(String(e)));
  await p.goto(url, { waitUntil: 'networkidle' }); await p.waitForSelector('[data-testid="stMain"] h1', { timeout: 120000 }); await idle(p);
  res.checks.push({ tag: 'inicial', ...(await check(p)) });
  await p.screenshot({ path: path.join(out, 'I_inicial.png'), fullPage: true });
  const sel = p.locator('[data-testid="stSelectbox"]').first();
  for (const [i, name] of [[4, 'incidente'], [2, 'caida_dia']]) {
    await sel.click(); await p.locator('[data-testid="stSelectboxVirtualDropdown"] li, [role="option"]').nth(i).click(); await idle(p);
    res.checks.push({ tag: name, ...(await check(p)) });
    await p.screenshot({ path: path.join(out, `I_${name}.png`), fullPage: true });
  }
  const m = await b.newPage({ ...devices['iPhone 13'] });
  await m.goto(url, { waitUntil: 'networkidle' }); await m.waitForSelector('[data-testid="stMain"] h1', { timeout: 120000 }); await idle(m);
  res.checks.push({ tag: 'movil', ...(await check(m)) });
  await m.screenshot({ path: path.join(out, 'I_movil.png'), fullPage: true });
  res.errors = errs; fs.writeFileSync(path.join(out, 'investigacion.json'), JSON.stringify(res, null, 1));
  console.log(JSON.stringify(res)); await b.close();
})().catch((e) => { console.error('FALLO', e); process.exit(1); });
