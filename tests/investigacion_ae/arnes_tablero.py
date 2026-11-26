"""Arnés local para validar la página del tablero: un sustituto mínimo de `dash` + tokens de tema + datos en línea.
Uso: D3_JS=<ruta a d3.min.js 7.9.0> python tests/investigacion_ae/arnes_tablero.py <index.html> <dir_datos> <salida.html> [dark]
Luego: node tests/investigacion_ae/validar_tablero.js <salida.html> <prefijo_capturas>  (errores, desbordes, vacíos, capturas)"""
import json
import sys
from pathlib import Path

page, data_dir, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
dark = len(sys.argv) > 4 and sys.argv[4] == "dark"
import os
d3 = Path(os.environ.get("D3_JS", Path(__file__).resolve().parents[1] / "node_modules" / "d3" / "dist" / "d3.min.js")).read_text()  # d3@7.9.0 (npm)
data = {}
raw = {f.stem: f.read_text(encoding="utf-8") for f in data_dir.glob("*.json")}  # se parsean en el navegador con JSON.parse
light = dict(fg="#1f1e1d", muted="#6b6a66", bg="#ffffff", panel="#faf9f7", line="#e3e1dc", grid="#ecebe7", axis="#c9c7c1",
             ref="#8a8883", pop="#ffffff", txt2="#5f5e5a", c=["#2a78d6", "#eb6834", "#1baf7a", "#a35fd6", "#d6a12a"])
darkp = dict(fg="#f2f1ee", muted="#a8a6a1", bg="#1a1a19", panel="#232322", line="#3a3936", grid="#2e2d2b", axis="#55534f",
             ref="#8f8d88", pop="#2a2a28", txt2="#b5b3ad", c=["#3987e5", "#d95926", "#199e70", "#b07ce8", "#c98500"])
p = darkp if dark else light
tokens = f"""
:root{{--color-fg:{p['fg']};--color-fg-muted:{p['muted']};--color-bg:{p['bg']};--color-panel:{p['panel']};
--color-border-line:{p['line']};--color-ok:#1d8a4a;--color-warn:#b7791f;--color-bad:#c2392b;
--font-anthropic-sans:system-ui,sans-serif;--font-anthropic-serif:Georgia,serif;--cds-font-size-caption:12px;
--cds-font-size-body:14px;--cds-font-size-heading:18px;--cds-font-size-title:28px;--cds-font-weight-medium:600;
--cds-gap-xs:4px;--cds-gap-sm:8px;--cds-gap-md:12px;--cds-gap-lg:16px;--cds-gap-xl:28px;--cds-pad-xs:4px;--cds-pad-sm:8px;
--cds-pad-md:12px;--cds-pad-lg:16px;--cds-pad-xl:24px;--cds-radius:8px;--cds-chart-muted:{p['axis']};--cds-chart-status-critical:#d03b3b;
--cds-chart-grid:{p['grid']};--cds-chart-axis:{p['axis']};--cds-chart-reference:{p['ref']};--cds-surface-popover:{p['pop']};
--cds-border:{p['line']};--cds-shadow-popover:0 2px 8px rgba(0,0,0,.2);--cds-text-secondary:{p['txt2']};
{''.join(f'--c{i}:{c};' for i, c in enumerate(p['c']))}}}
body{{margin:0;background:{p['bg']};color:{p['fg']};font:14px system-ui}} #dash-root{{padding:16px}}
.dash-skeleton{{background:{p['grid']}}}"""
mock = """
window.addEventListener('error',e=>__ERR.push(String(e.message)));
const __draws={}, __ondata=[];
window.dash={colors:[0,1,2,3,4].map(i=>`var(--c${i})`),
 data(id){const r=__DATA[id];return r?{status:'ok',data:r,columns:Object.keys(r[0]||{})}:{status:'missing',data:[]};},
 draw(name,fn){__draws[name]=fn;},onData(fn){__ondata.push(fn);},params(){return {};},setParams(){},resetParams(){},setLink(){},
 calc(){},loader(){},refresh(){}};
window.__runDraws=function(){__ondata.forEach(f=>f());for(const [n,fn] of Object.entries(__draws)){
 document.querySelectorAll(`[data-draw="${n}"]`).forEach(el=>{try{(typeof fn==='function'?fn:fn.update||fn.mount)(el)}catch(e){__ERR.push(n+': '+e.message)}});}};
const __origDraw=dash.draw; dash.draw=function(name,fn){ if(fn===undefined){document.querySelectorAll(`[data-draw="${name}"]`).forEach(el=>{try{__draws[name](el)}catch(e){__ERR.push(name+': '+e.message)}});return;} __origDraw(name,fn); };
"""
html = page.read_text(encoding="utf-8")
out.write_text(f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<style>{tokens}</style><script>{d3}</script><script>window.__ERR=[];window.__DATA={{}};const __RAW={json.dumps(raw, ensure_ascii=False)};
for(const [k,t] of Object.entries(__RAW)){{try{{const j=JSON.parse(t);window.__DATA[k]=(j&&j.rows)?j.rows:j;}}catch(e){{window.__ERR.push('JSON inválido en '+k+': '+e.message);}}}}
{mock}</script></head>
<body><div id="dash-root"></div><template id="pg">{html}</template>
<script>
const root=document.getElementById('dash-root');root.innerHTML=document.getElementById('pg').innerHTML;
root.querySelectorAll('script').forEach(s=>{{const n=document.createElement('script');n.textContent='(function(){{'+s.textContent+'\\n}})();';s.replaceWith(n);}});
requestAnimationFrame(()=>{{__runDraws();window.__READY=true;}});
window.addEventListener('resize',()=>__runDraws());
</script></body></html>""", encoding="utf-8")
print("ok", out, sorted(raw))
