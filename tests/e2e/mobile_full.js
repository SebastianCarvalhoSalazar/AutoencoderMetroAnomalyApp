// Capturas móviles completas (390 px de ancho, ventana alta para mostrar el contenido del contenedor interno).
// Uso: node tests/e2e/mobile_full.js <url> <archivo_xlsx> <dir_capturas>
const { chromium } = require('playwright');
const [url, file, shots] = process.argv.slice(2);
(async () => {
  const b = await chromium.launch();
  const c = await b.newContext({ viewport: { width: 390, height: 4200 }, deviceScaleFactor: 1, isMobile: true, hasTouch: true });
  const p = await c.newPage(); const errs = [];
  p.on('pageerror', e => errs.push(String(e))); p.on('console', m => { if (m.type() === 'error') errs.push(m.text()); });
  await p.goto(url, { waitUntil: 'networkidle' });
  await p.waitForSelector('[data-testid="stMain"] [data-testid="stAlert"]', { timeout: 120000 }); await p.waitForTimeout(2500);
  const ov = await p.evaluate(() => ({ sw: document.documentElement.scrollWidth, cw: document.documentElement.clientWidth }));
  await p.screenshot({ path: `${shots}/M03_movil_historico_completo.png` });
  await p.getByRole('tab', { name: /Cargar archivo/ }).click(); await p.waitForTimeout(800);
  await p.setInputFiles('input[type=file]', file);
  await p.waitForSelector('text=/día\\(s\\) evaluados/', { timeout: 120000 }); await p.waitForTimeout(3000);
  await p.screenshot({ path: `${shots}/M04_movil_carga_completa.png` });
  console.log(JSON.stringify({ overflow: ov, errores: errs }));
  await b.close();
})();
