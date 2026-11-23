// E2E de la app (Playwright). Uso: node tests/e2e/app.e2e.js <url> <valido.xlsx> <invalido.csv> <dir_capturas>
const { chromium } = require('playwright');
const [url, good, bad, shots] = process.argv.slice(2);
const fail = (m) => { console.error('FALLO: ' + m); process.exit(1); };
(async () => {
  const b = await chromium.launch();
  const p = await b.newPage({ viewport: { width: 1400, height: 1500 } });
  await p.goto(url, { waitUntil: 'networkidle' });
  // Nota: 'text=' no distingue mayúsculas (coincidía con "anomalías" del título); se espera la alerta real.
  await p.waitForSelector('[data-testid="stMain"] [data-testid="stAlert"]', { timeout: 120000 });
  await p.waitForSelector('[data-testid="stVegaLiteChart"], .vega-embed', { timeout: 60000 });
  if (await p.locator('[data-testid="stException"]').count()) fail('excepción en carga inicial');
  await p.screenshot({ path: `${shots}/e2e_1_historico.png`, fullPage: true });

  // Latencia (A5): cambiar la fecha varias veces con caché caliente
  const lat = [];
  const dates = ['2024/03/06', '2023/10/29', '2024/01/11', '2023/12/14', '2024/05/15', '2023/08/28'];
  // Corrección F9: fill() no aplica la fecha en el campo con máscara (la versión anterior medía reejecuciones
  // sin cambio). Se escribe carácter a carácter y se espera a que el encabezado muestre la nueva fecha.
  for (const d of dates) {
    const inp = p.locator('[data-testid="stDateInput"] input');
    const cur = await inp.inputValue();
    const [y, mo, da] = d.split('/');
    const val = /^\d{4}\//.test(cur) ? d : `${da}/${mo}/${y}`;
    await inp.click(); await inp.press('Control+A'); await inp.pressSequentially(val, { delay: 10 });
    const t0 = Date.now();
    await inp.press('Enter');
    const iso = `${y}-${mo}-${da}`;
    await p.waitForSelector(`[data-testid="stMain"] strong:has-text("${iso}")`, { timeout: 30000 });
    await p.waitForFunction(() => !document.querySelector('[data-testid="stStatusWidget"]'), null, { timeout: 30000 });
    lat.push(Date.now() - t0);
    await p.keyboard.press('Escape');
  }
  lat.sort((a, c) => a - c);
  const p95 = lat[Math.min(lat.length - 1, Math.ceil(0.95 * lat.length) - 1)];
  console.log('LATENCIAS_MS', JSON.stringify(lat), 'P95', p95);

  // Carga de archivo válido con interrupción inyectada
  await p.getByRole('tab', { name: /Cargar archivo/ }).click();
  await p.setInputFiles('input[type=file]', good);
  await p.waitForSelector('text=/día\\(s\\) evaluados/', { timeout: 120000 });
  const info = await p.locator('[data-testid="stAlert"]').allInnerTexts();
  console.log('INFO_CARGA', JSON.stringify(info.filter(t => t.includes('evaluados'))));
  if (!info.some(t => /evaluados · [1-9]\d* anomal/.test(t))) fail('no se detectó la anomalía inyectada');
  if (!(await p.locator('text=2024-06-12').count())) fail('no aparece el día inyectado');
  await p.screenshot({ path: `${shots}/e2e_2_carga_valida.png`, fullPage: true });

  // Archivo inválido: mensaje controlado, sin traza
  await p.setInputFiles('input[type=file]', bad);
  await p.waitForSelector('text=/❌/', { timeout: 60000 });
  if (await p.locator('[data-testid="stException"]').count()) fail('traza visible con archivo inválido');
  console.log('ERROR_CONTROLADO', JSON.stringify((await p.locator('[data-testid="stAlert"]').allInnerTexts()).filter(t => t.includes('❌'))));
  await p.screenshot({ path: `${shots}/e2e_3_carga_invalida.png`, fullPage: true });
  await b.close();
  if (p95 >= 2000) fail(`latencia p95 ${p95} ms >= 2000 ms`);
  console.log('E2E OK');
})().catch(e => fail(e.message));
