// Uso: node shot.js <harness.html> <salida_prefijo>
const { chromium } = require('playwright');
const [file, out] = process.argv.slice(2);
(async () => {
  const b = await chromium.launch();
  const res = {};
  for (const [tag, w] of [['desk', 1280], ['phone', 400]]) {
    const p = await b.newPage({ viewport: { width: w, height: 900 }, deviceScaleFactor: 1 });
    const logs = [];
    p.on('console', (m) => { if (m.type() === 'error') logs.push(m.text()); });
    p.on('pageerror', (e) => logs.push(String(e)));
    await p.goto('file://' + file);
    await p.waitForFunction(() => window.__READY === true, null, { timeout: 30000 });
    await p.waitForTimeout(500);
    const info = await p.evaluate(() => {
      const vw = document.documentElement.clientWidth;
      const over = [];
      document.querySelectorAll('#dash-root *').forEach((el) => {
        const r = el.getBoundingClientRect();
        if (r.width > 0 && r.right > vw + 1 && !el.closest('.ae-tablewrap')) over.push((el.tagName + '.' + (el.className.baseVal ?? el.className)).slice(0, 60));
      });
      const empty = [...document.querySelectorAll('[data-draw]')].filter((el) => !el.textContent.trim() && !el.querySelector('svg,li,tr,option')).map((el) => el.dataset.draw);
      const dashes = [...document.querySelectorAll('[data-draw]')].filter((el) => el.textContent.trim() === '—').map((el) => el.dataset.draw);
      return { errors: window.__ERR, overflowing: [...new Set(over)].slice(0, 15), scrollW: document.documentElement.scrollWidth, vw, empty, dashes };
    });
    info.console = logs;
    res[tag] = info;
    await p.screenshot({ path: `${out}_${tag}.png`, fullPage: true });
    if (tag === 'desk' && await p.locator('#ej-select').count()) {
      for (const v of ['caida_horas', 'caida_dia', 'hora_duplicada', 'incidente_real']) {
        await p.selectOption('#ej-select', v); await p.waitForTimeout(300);
        await p.locator('#ejemplos').screenshot({ path: `${out}_ej_${v}.png` });
      }
      info.errors_after_select = await p.evaluate(() => window.__ERR);
    }
    await p.close();
  }
  console.log(JSON.stringify(res, null, 1));
  await b.close();
})();
