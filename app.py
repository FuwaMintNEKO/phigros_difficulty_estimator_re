"""Local web UI for the difficulty estimator.

    python app.py            # http://127.0.0.1:5000

Drop one or many charts (`.json` / `.pec` / `.pez` / a `.zip` chart package).
Each card carries its own speed slider, the chart's own metadata, the note-type
mix and the five player-facing categories behind the constant.
"""

from __future__ import annotations

import os
import tempfile

from flask import Flask, Response, jsonify, request

from rating.compose import CATEGORIES
from rating.predictor import Predictor

app = Flask(__name__)
predictor = Predictor()

SUFFIXES = ('.json', '.pec', '.pez', '.zip')

PAGE = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Phigros 定数估算</title>
<style>
:root{
  --bg:#0b0e14;--bg2:#10151e;--panel:rgba(22,28,39,.82);--panel-solid:#161c27;
  --line:rgba(255,255,255,.07);--line2:rgba(255,255,255,.13);
  --text:#e8eef6;--dim:#8794a8;--faint:#5d6a7e;
  --accent:#ffb454;--warn:#ff7b72;
  --tap:#5ec8ff;--hold:#ffb454;--drag:#4ade80;--flick:#c297ff;
  --radius:18px;
}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{margin:0;color:var(--text);min-height:100vh;
  font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;
  background:
    radial-gradient(900px 480px at 12% -8%,rgba(94,200,255,.10),transparent 60%),
    radial-gradient(900px 520px at 92% -12%,rgba(255,123,114,.10),transparent 60%),
    linear-gradient(180deg,var(--bg2),var(--bg) 480px);
  background-attachment:fixed}
main{width:min(1060px,100%);margin:0 auto;padding:52px 22px 80px}

header{display:flex;align-items:flex-end;justify-content:space-between;gap:20px;margin-bottom:26px;flex-wrap:wrap}
h1{margin:0;font-size:28px;letter-spacing:.4px;font-weight:700;
  background:linear-gradient(90deg,#fff 0%,#ffd9a0 55%,#ff9d8f 100%);-webkit-background-clip:text;background-clip:text;color:transparent}
.sub{color:var(--dim);font-size:13.5px;margin-top:4px}
.legend-scale{display:flex;align-items:center;gap:9px;font-size:11.5px;color:var(--faint)}
.legend-scale .bar{width:150px;height:6px;border-radius:99px;
  background:linear-gradient(90deg,hsl(120 72% 55%),hsl(60 72% 55%),hsl(0 72% 55%))}

#drop{position:relative;border:1.5px dashed var(--line2);border-radius:var(--radius);padding:34px 20px;
  text-align:center;background:var(--panel);backdrop-filter:blur(8px);cursor:pointer;transition:.2s}
#drop:hover{border-color:rgba(255,180,84,.55);background:rgba(28,35,48,.9)}
#drop.hot{border-color:var(--accent);background:rgba(255,180,84,.07);transform:translateY(-2px);
  box-shadow:0 12px 40px -12px rgba(255,180,84,.35)}
#drop .ico{font-size:26px;line-height:1;margin-bottom:8px;opacity:.9}
#drop b{color:var(--accent);font-weight:600}
#drop small{display:block;color:var(--faint);margin-top:6px;font-size:12px}

#toolbar{display:flex;gap:10px;align-items:center;margin:22px 0 2px;flex-wrap:wrap}
.tool{background:var(--panel);color:var(--dim);border:1px solid var(--line);border-radius:10px;
  padding:7px 14px;font-size:12.5px;cursor:pointer;transition:.15s;font-family:inherit}
.tool:hover{color:var(--text);border-color:var(--line2)}
.tool.danger:hover{color:var(--warn);border-color:rgba(255,123,114,.5)}
.spacer{flex:1}
#count{color:var(--faint);font-size:12.5px}

.card{position:relative;background:var(--panel);backdrop-filter:blur(10px);border:1px solid var(--line);
  border-radius:var(--radius);padding:20px 24px 14px;margin-top:16px;overflow:hidden;
  animation:rise .38s cubic-bezier(.2,.8,.2,1);transition:border-color .2s,transform .2s}
.card:hover{border-color:var(--line2)}
.card::before{content:"";position:absolute;inset:0 auto 0 0;width:3px;background:var(--tint,var(--accent));opacity:.9}
@keyframes rise{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:none}}

.row{display:grid;grid-template-columns:150px minmax(190px,1fr) 150px minmax(250px,1.15fr);gap:22px;align-items:center}
@media(max-width:940px){.row{grid-template-columns:150px 1fr}.pie,.bars{grid-column:1/-1}}
@media(max-width:520px){.row{grid-template-columns:1fr}}

.gauge{position:relative;width:132px;height:132px}
.gauge svg{display:block}
.gauge .num{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center}
.gauge .v{font-size:35px;font-weight:700;line-height:1;font-variant-numeric:tabular-nums;letter-spacing:-.5px}
.gauge .c{font-size:11.5px;color:var(--dim);margin-top:6px;font-variant-numeric:tabular-nums}
.track-ring{stroke:rgba(255,255,255,.07)}
.arc{transition:stroke-dashoffset .6s cubic-bezier(.2,.8,.2,1),stroke .4s}

.title{font-size:17px;font-weight:650;margin:0 0 4px;word-break:break-word}
.meta{color:var(--dim);font-size:12.5px;line-height:1.75}
.meta .chip{display:inline-block;padding:0 8px;margin:0 6px 4px 0;border-radius:99px;
  background:rgba(255,255,255,.05);border:1px solid var(--line)}

.pie{display:flex;align-items:center;gap:12px}
.legend{font-size:11.5px;color:var(--dim);line-height:1.8;white-space:nowrap}
.legend i{display:inline-block;width:8px;height:8px;border-radius:3px;margin-right:6px}
.legend b{font-weight:500;color:var(--text);font-variant-numeric:tabular-nums}

.bars .item{margin:9px 0}
.bars .h{display:flex;justify-content:space-between;font-size:12.5px;margin-bottom:5px;color:var(--dim)}
.bars .h span:last-child{font-variant-numeric:tabular-nums;color:var(--text)}
.track{height:8px;background:rgba(255,255,255,.06);border-radius:99px;overflow:hidden}
.fill{height:100%;border-radius:99px;transition:width .5s cubic-bezier(.2,.8,.2,1)}

.ctrl{display:flex;align-items:center;gap:12px;margin-top:14px;padding-top:11px;border-top:1px solid var(--line);
  font-size:12.5px;color:var(--dim)}
.ctrl input[type=range]{flex:1;accent-color:var(--accent);height:4px}
.ctrl .sp{min-width:48px;text-align:right;color:var(--text);font-variant-numeric:tabular-nums}
.ctrl .close{background:none;border:none;color:var(--faint);font-size:19px;line-height:1;cursor:pointer;
  padding:2px 6px;border-radius:7px;transition:.15s}
.ctrl .close:hover{color:var(--warn);background:rgba(255,123,114,.1)}

.empty{margin-top:34px;text-align:center;color:var(--faint);font-size:13px}
#toast{position:fixed;left:50%;bottom:26px;transform:translate(-50%,16px);opacity:0;pointer-events:none;
  background:#2a1618;border:1px solid rgba(255,123,114,.4);color:#ffb3ad;padding:10px 16px;border-radius:12px;
  font-size:13px;max-width:min(92vw,560px);white-space:pre-wrap;transition:.25s;z-index:9}
#toast.show{opacity:1;transform:translate(-50%,0)}
.note{margin-top:30px;color:var(--faint);font-size:12px;line-height:1.8;border-top:1px solid var(--line);padding-top:16px}
.note b{color:var(--dim);font-weight:600}
.hidden{display:none}
</style></head><body><main>

<header>
  <div>
    <h1>Phigros 定数估算</h1>
    <div class="sub">官谱 / RPE / PEC / 谱包 · 可批量导入</div>
  </div>
  <div class="legend-scale"><span>0</span><div class="bar"></div><span>18</span></div>
</header>

<div id="drop">
  <div class="ico">⬇</div>
  <b>拖入谱面文件</b>，或点击选择
  <small>.json · .pec · .pez · .zip（含 chart.json 的谱包）· 可多选</small>
</div>
<input id="file" type="file" class="hidden" multiple accept=".json,.pec,.pez,.zip">

<div id="toolbar" class="hidden">
  <button class="tool" id="sortBtn">按定数降序 ↓</button>
  <button class="tool danger" id="clearBtn">清空</button>
  <div class="spacer"></div>
  <span id="count"></span>
</div>
<div id="cards"></div>
<div class="empty" id="empty">还没有谱面，拖一张进来试试</div>
<div id="toast"></div>

<div class="note">
  <b>难度构成</b>：把某一类的全部特征替换成训练集里的中位谱重新预测，前后差值就是这一类的影响，
  五类归一化后相加 100%。<b>变速</b>按 Phira 语义（整谱时间压缩）重算，只影响该张谱面；
  多押是"50ms 窗口内整数计数"，原始曲线会有台阶，所以显示值取采样后的单调包络。
</div>
</main>
<script>
const $=id=>document.getElementById(id);
const drop=$('drop'), fileInput=$('file'), wrap=$('cards'), toolbar=$('toolbar');
const sortBtn=$('sortBtn'), clearBtn=$('clearBtn'), countEl=$('count'), emptyEl=$('empty'), toast=$('toast');
const COLORS={volume:'#5ec8ff',burst:'#ffb454',chords:'#c297ff',movement:'#4ade80',stamina:'#ff7b72'};
const PIE={tap:'#5ec8ff',hold:'#ffb454',drag:'#4ade80',flick:'#c297ff'};
const PIE_NAME={tap:'Tap',hold:'Hold',drag:'Drag',flick:'Flick'};
let state=[], nextKey=1, descending=true, toastTimer=null;
const timers={};

const hue=v=>120*(1-Math.min(Math.max(v/18,0),1));          // 0 -> green(120), 18 -> red(0)
const scoreColor=v=>`hsl(${hue(v).toFixed(0)} 72% 58%)`;

function showError(msg){
  toast.textContent=msg; toast.classList.add('show');
  clearTimeout(toastTimer); toastTimer=setTimeout(()=>toast.classList.remove('show'),5200);
}

drop.onclick=()=>fileInput.click();
fileInput.onchange=e=>{ if(e.target.files.length) sendAll([...e.target.files]); fileInput.value=''; };
['dragenter','dragover'].forEach(t=>drop.addEventListener(t,e=>{e.preventDefault();drop.classList.add('hot')}));
['dragleave','drop'].forEach(t=>drop.addEventListener(t,e=>{e.preventDefault();drop.classList.remove('hot')}));
drop.addEventListener('drop',e=>{ if(e.dataTransfer.files.length) sendAll([...e.dataTransfer.files]); });
window.addEventListener('dragover',e=>e.preventDefault());
window.addEventListener('drop',e=>e.preventDefault());

async function sendAll(files){
  const label=drop.querySelector('b'); const old=label.textContent;
  const errors=[];
  let done=0;
  for(const f of files){
    label.textContent=`解析中 ${++done}/${files.length}…`;
    const body=new FormData(); body.append('chart',f); body.append('speed','1');
    try{
      const res=await fetch('/api/predict',{method:'POST',body});
      const data=await res.json();
      if(!res.ok) throw new Error(data.error||'解析失败');
      state.push({key:nextKey++, path:data.path, speed:1, data});
      render();
    }catch(e){ errors.push(f.name+'：'+e.message); }
  }
  label.textContent=old;
  if(errors.length) showError(errors.join('\\n'));
}

function metaChips(d){
  const m=d.meta||{}, out=[];
  const sec=d.stats.duration, mm=Math.floor(sec/60), ss=Math.round(sec%60);
  out.push(`${mm}:${String(ss).padStart(2,'0')}`);
  if(m.bpm_min&&m.bpm_max){
    out.push(Math.abs(m.bpm_max-m.bpm_min)<0.01 ? `BPM ${m.bpm_min.toFixed(0)}`
      : `BPM ${m.bpm_min.toFixed(0)}~${m.bpm_max.toFixed(0)}`);
  }
  if(m.composer) out.push(m.composer);
  if(m.charter) out.push('谱师 '+m.charter);
  if(m.level) out.push(m.level);
  out.push(d.stats.notes+' 音符');
  return out;
}

function gaugeSvg(v){
  const r=54, c=2*Math.PI*r, frac=Math.min(Math.max(v/18,0),1);
  return `<svg width="132" height="132" viewBox="0 0 132 132">
    <circle class="track-ring" cx="66" cy="66" r="${r}" fill="none" stroke-width="9"></circle>
    <circle class="arc" cx="66" cy="66" r="${r}" fill="none" stroke-width="9" stroke-linecap="round"
      stroke="${scoreColor(v)}" stroke-dasharray="${c}" stroke-dashoffset="${c*(1-frac)}"
      transform="rotate(-90 66 66)"></circle></svg>`;
}

function pieSvg(mix){
  const total=(mix.tap+mix.hold+mix.drag+mix.flick)||1;
  const r=40, c=2*Math.PI*r; let off=0, arcs='';
  for(const [k,v] of Object.entries(mix)){
    const len=c*(v/total); if(len<=0) continue;
    arcs+=`<circle cx="50" cy="50" r="${r}" fill="none" stroke="${PIE[k]}" stroke-width="15"
      stroke-dasharray="${Math.max(len-1.4,0.1)} ${c-len+1.4}" stroke-dashoffset="${-off}"
      transform="rotate(-90 50 50)"></circle>`;
    off+=len;
  }
  const legend=Object.entries(mix).map(([k,v])=>
    `<div><i style="background:${PIE[k]}"></i>${PIE_NAME[k]} <b>${(100*v/total).toFixed(0)}%</b></div>`).join('');
  return `<svg width="100" height="100" viewBox="0 0 100 100">${arcs}</svg><div class="legend">${legend}</div>`;
}

function barsHtml(d){
  return Object.entries(d.categories).map(([k,label])=>{
    const p=d.percent[k]||0, w=Math.max(2,Math.min(100,Math.abs(p)));
    return `<div class="item"><div class="h"><span>${label}</span><span>${p.toFixed(0)}%</span></div>
      <div class="track"><div class="fill" style="width:${w}%;background:${COLORS[k]};opacity:${p<0?0.35:1}"></div></div></div>`;
  }).join('');
}

function bodyHtml(d){
  const chips=metaChips(d).map(t=>`<span class="chip">${esc(t)}</span>`).join('');
  return `<div class="row">
    <div class="gauge">${gaugeSvg(d.official)}
      <div class="num"><div class="v" style="color:${scoreColor(d.official)}">${d.official.toFixed(2)}</div>
      <div class="c">社区 ${d.community.toFixed(2)}</div></div></div>
    <div><h3 class="title">${esc(d.meta.title||d.file)}</h3><div class="meta">${chips}</div></div>
    <div class="pie">${pieSvg(d.stats.mix)}</div>
    <div class="bars">${barsHtml(d)}</div>
  </div>`;
}

function esc(s){ return String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }

function ctrlHtml(item){
  return `<div class="ctrl"><span>变速</span>
    <input type="range" min="0.6" max="2" step="0.05" value="${item.speed}" data-key="${item.key}">
    <span class="sp">${item.speed.toFixed(2)}x</span>
    <button class="close" data-del="${item.key}" title="移除这张谱面" aria-label="移除">×</button></div>`;
}

function render(){
  const has=state.length>0;
  toolbar.classList.toggle('hidden',!has);
  emptyEl.classList.toggle('hidden',has);
  countEl.textContent = has ? `${state.length} 张` : '';
  sortBtn.textContent = descending ? '按定数降序 ↓' : '按定数升序 ↑';
  const order=state.slice().sort((a,b)=>descending ? b.data.official-a.data.official
                                                     : a.data.official-b.data.official);
  wrap.innerHTML='';
  for(const item of order){
    const el=document.createElement('div');
    el.className='card'; el.id='c'+item.key;
    el.style.setProperty('--tint',scoreColor(item.data.official));
    el.innerHTML=`<div class="body">${bodyHtml(item.data)}</div>`+ctrlHtml(item);
    wrap.appendChild(el);
  }
}

wrap.addEventListener('input',e=>{
  const input=e.target.closest('input[type=range]'); if(!input) return;
  const key=Number(input.dataset.key), item=state.find(x=>x.key===key); if(!item) return;
  item.speed=Number(input.value);
  document.querySelector('#c'+key+' .sp').textContent=item.speed.toFixed(2)+'x';
  clearTimeout(timers[key]);
  timers[key]=setTimeout(()=>rescore(item),240);
});

wrap.addEventListener('click',e=>{
  const del=e.target.closest('[data-del]'); if(!del) return;
  const key=Number(del.dataset.del);
  state=state.filter(x=>x.key!==key);
  render();
});

async function rescore(item){
  const body=new FormData(); body.append('path',item.path); body.append('speed',item.speed);
  try{
    const res=await fetch('/api/predict',{method:'POST',body});
    const data=await res.json();
    if(!res.ok) throw new Error(data.error||'重算失败');
    item.data=data;
    const card=$('c'+item.key);
    if(card){
      card.querySelector('.body').innerHTML=bodyHtml(data);
      card.style.setProperty('--tint',scoreColor(data.official));
    }
  }catch(e){ showError(e.message); }
}

sortBtn.onclick=()=>{ descending=!descending; render(); };
clearBtn.onclick=()=>{ state=[]; render(); };
render();
</script></body></html>"""


def _predict_file(path: str, speed: float, display: str = '') -> dict:
    payload = predictor.explain(path, speed, monotone=True)
    payload['path'] = path
    if display:
        payload['file'] = display
        if not payload['meta'].get('title'):
            payload['meta']['title'] = os.path.splitext(display)[0]
    return payload


@app.route('/')
def index():
    # served verbatim: the page carries CSS/JS braces that Jinja would try to parse
    return Response(PAGE, mimetype='text/html; charset=utf-8')


@app.route('/api/predict', methods=['POST'])
def api_predict():
    speed = float(request.form.get('speed', 1.0) or 1.0)
    if 'chart' in request.files:
        upload = request.files['chart']
        suffix = os.path.splitext(upload.filename)[1].lower()
        if suffix not in SUFFIXES:
            return jsonify({'error': f'不支持的文件类型 {suffix or "（无扩展名）"}'}), 400
        handle = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        upload.save(handle.name)
        handle.close()
        try:
            return jsonify(_predict_file(handle.name, speed, upload.filename))
        except Exception as exc:  # noqa: BLE001
            return jsonify({'error': str(exc)}), 400
    path = request.form.get('path', '').strip()
    if path and os.path.exists(path):
        try:
            return jsonify(_predict_file(path, speed))
        except Exception as exc:  # noqa: BLE001
            return jsonify({'error': str(exc)}), 400
    return jsonify({'error': '没有收到谱面文件'}), 400


if __name__ == '__main__':
    # 5000 is commonly taken (Style-Bert-VITS2, many dev servers); stay off it.
    port = int(os.environ.get('PORT', 5050))
    print(f'Phigros 定数估算  ->  http://127.0.0.1:{port}')
    app.run(host='127.0.0.1', port=port, debug=False, threaded=True)
