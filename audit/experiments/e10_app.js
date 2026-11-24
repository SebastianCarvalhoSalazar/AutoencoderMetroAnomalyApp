const { chromium } = require('playwright');
(async () => {
  const b = await chromium.launch(); 
  for (const f of ['x_test_normal_1.pkl','x_test_normal_150.pkl','x_test_outlier_1.pkl']) {
    const p = await b.newPage({viewport:{width:1300,height:1100}});
    await p.goto('http://localhost:8501', {waitUntil:'networkidle'});
    await p.waitForSelector('input[type=file]', {state:'attached', timeout:60000});
    const t0=Date.now();
    await p.setInputFiles('input[type=file]', '/home/user/AutoencoderMetroAnomalyApp/dev/data_test/'+f);
    await p.waitForSelector('text=corresponde a un sujeto', {timeout:120000});
    await p.waitForTimeout(2500);
    const txt = await p.locator('[data-testid="stAlert"]').allInnerTexts();
    console.log(f, '=>', JSON.stringify(txt), 'latency_ms', Date.now()-t0);
    await p.screenshot({path: `app_${f}.png`, fullPage:true});
    await p.close();
  }
  await b.close();
})();
