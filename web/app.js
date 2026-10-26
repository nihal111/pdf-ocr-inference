'use strict';
const content = document.querySelector('#content');
let state, stage = 'render', generation = 0, currentProfile;
const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const num = (v, digits=1) => typeof v === 'number' && Number.isFinite(v) ? v.toLocaleString(undefined, {maximumFractionDigits:digits}) : '—';
const duration = v => typeof v === 'number' && Number.isFinite(v) ? `${num(v/1000,v>=1000?2:v>=1?4:6)} s` : '—';
function suiteRuntime(v) {
  const runtime=v.suite_runtime;
  if(!runtime)return '<span class="muted" title="Suite wall time was not recorded">—</span>';
  const seconds=Math.round(runtime.seconds);
  const label=seconds>=3600?`${Math.floor(seconds/3600)}h ${Math.floor(seconds%3600/60)}m ${seconds%60}s`:seconds>=60?`${Math.floor(seconds/60)}m ${String(seconds%60).padStart(2,'0')}s`:`${seconds}s`;
  return `<span title="${esc(runtime.scope||'Full evaluation wall time')}${runtime.source==='sweep_log_rounded_seconds'?' Recorded in the sweep log to the nearest second.':''}">${label}</span>`;
}
const statusLabel = value => value === 'awaiting_review' ? 'Awaiting review' : String(value ?? '').replaceAll('_',' ');
const badge = (value, label=statusLabel(value)) => `<span class="badge ${esc(value)}">${esc(label)}</span>`;
const pretty = value => `<pre class="json"><code>${esc(JSON.stringify(value, null, 2))}</code></pre>`;
const empty = (title, text, action='', compact=false) => `<div class="empty ${compact?'compact':''}"><div class="empty-icon">≋</div><h3>${esc(title)}</h3><p>${esc(text)}</p>${action}</div>`;
const heading = (eyebrow, title, description, right='') => `<div class="heading-row"><div><div class="eyebrow">${esc(eyebrow)}</div><h1>${esc(title)}</h1><p class="lead">${esc(description)}</p></div>${right}</div>`;
const metric = (label, value, note, text=false) => `<div class="metric"><div class="metric-label">${esc(label)}</div><div class="metric-value ${text?'text':''}">${esc(value)}</div><div class="metric-note">${esc(note)}</div></div>`;
const detail = (title, value) => `<details><summary>${esc(title)}</summary>${pretty(value)}</details>`;
const sectionHead = (title, desc='', action='') => `<div class="section-head"><div><h2>${esc(title)}</h2>${desc?`<p>${esc(desc)}</p>`:''}</div>${action}</div>`;
const encodePath = path => path.split('/').map(encodeURIComponent).join('/');
const artifactLink = a => state.asset_urls?.[a.path]||`artifacts/${encodePath(a.path)}`;
const date = value => value ? String(value).replace('T',' · ').replace(/\.\d+.*$/,'').replace('+00:00',' UTC') : 'Date unavailable';
async function loadJSON(path) { const r = await fetch(path); if (!r.ok) throw new Error(`Could not load ${path} (${r.status})`); return r.json(); }
function toast(text) { const el=document.querySelector('#toast'); el.textContent=text; el.classList.add('visible'); setTimeout(()=>el.classList.remove('visible'),2200); }
function nav() {
  const route=location.hash || '#/';
  const link=(url,label,icon,guide=false)=>`<a class="nav-link ${guide?'guide':''} ${route===url?'active':''}" href="${url}" ${route===url?'aria-current="page"':''}><span class="nav-icon" aria-hidden="true">${icon}</span>${esc(label)}</a>`;
  document.querySelector('#navigation').innerHTML=`<div class="nav-label">WORKSPACE</div>${link('#/','Overview','◫')}${link('#/runs','Experiment runs','▤')}${link('#/datasets','PDF dataset','▥')}${link('#/journal','Experiment journal','▧')}<div class="nav-label">NOTES</div>${state.documents.filter(d=>d.kind==='guide').map(d=>link(`#/doc/${d.id}`,d.title,'·',true)).join('')}<div class="nav-label">RESULTS</div>${link('#/output/latest','PDF → Markdown','⇄')}${link('#/embeddings/latest','Embeddings','⠿')}`;
  if(route.startsWith('#/output/'))document.querySelector('a[href="#/output/latest"]').setAttribute('aria-current','page');
  if(route.startsWith('#/embeddings/'))document.querySelector('a[href="#/embeddings/latest"]').setAttribute('aria-current','page');
}
function stagePanel() {
  const cfg=state.defaults, d={
    render:{title:'Render each page',text:'Convert every PDF page into an RGB image. All pages are rendered before OCR begins.',values:{Library:'pypdfium2',Resolution:`${cfg.render.dpi} DPI`,Output:'One RGB image per page',Execution:'Local CPU'}},
    ocr:cfg.ocr.backend==='paddleocr'
      ? {title:'Parse each page with PaddleOCR',text:'PP-DocLayoutV3 finds regions and reading order. PaddleOCR-VL-1.6 recognizes text, tables and formulas, then assembles page Markdown. Four workers cooperate on the current PDF.',values:{Model:cfg.ocr.model.split('/').pop(),Backend:cfg.ocr.paddle_rec_backend==='vllm-server'?'Paddle layout + vLLM recognition':'Native PaddlePaddle','Layout detector':'PP-DocLayoutV3','GPU workers':cfg.ocr.gpu_workers,'Pages per GPU':cfg.ocr.page_batch_size,'Regions per batch':cfg.ocr.paddle_region_batch_size,'Token ceiling per region':cfg.ocr.max_new_tokens,'Completion check':'EOS required for every region'}}
      : {title:'Transcribe the page image',text:'GPU workers process pages from the same PDF, then restore their original order. Greedy decoding selects one token at a time.',values:{Model:cfg.ocr.model.split('/').pop(),Backend:cfg.ocr.backend==='hf'?'Hugging Face Transformers':cfg.ocr.backend,Precision:cfg.ocr.dtype,'GPU workers':cfg.ocr.gpu_workers,'Pages per GPU batch':cfg.ocr.page_batch_size,'Attention':cfg.ocr.attn_implementation,'Output token limit':cfg.ocr.max_new_tokens}},
    chunk:{title:'Build text chunks',text:'Pack blank-line Markdown blocks into smaller pieces, preserving headings for context. The token budget is a target, not an exact length.',values:{'Token target':cfg.chunk.max_tokens,Tokenizer:cfg.chunk.tokenizer.split('/').pop(),'Oversized blocks':'Split at token boundaries','Heading context':'Carried into later chunks'}},
    embed:{title:'Turn chunks into vectors',text:'Embed each piece independently, then normalize its vector. Chunk batch size is separate from OCR page batch size.',values:{Model:cfg.embed.model.split('/').pop(),Backend:cfg.embed.backend,Precision:cfg.embed.dtype,'Chunk batch':cfg.embed.batch_size}}
  }[stage];
  return `<div class="pipeline-tabs" role="tablist" aria-label="Pipeline stages">${['render','ocr','chunk','embed'].map((s,i)=>`<button class="stage-tab" type="button" role="tab" data-stage="${s}" aria-selected="${s===stage}"><span>0${i+1}</span><strong>${s==='ocr'?'OCR':s[0].toUpperCase()+s.slice(1)}</strong></button>`).join('')}</div><div class="stage-detail" role="tabpanel"><h3>${d.title}</h3><p>${d.text}</p><div class="kv-grid">${Object.entries(d.values).map(([k,v])=>`<div class="kv"><small>${esc(k)}</small><strong>${esc(v)}</strong></div>`).join('')}</div></div>`;
}
// Recorded milestones along the retained configuration's path. Repeated controls
// measure each change; they are not extra optimization steps.
const overviewMilestones=[
  ['d8a8a05ea1b9bc9e9268','vllm_batch_1','Batch 1 baseline','One region at a time on each page’s GPU.'],
  ['d8a8a05ea1b9bc9e9268','vllm_batch_4','Batch four OCR regions','Raise OCR region concurrency from one to four on each page’s assigned GPU; keep page placement fixed.','d8a8a05ea1b9bc9e9268','vllm_batch_1'],
  ['d8a8a05ea1b9bc9e9268','vllm_batch_4_all_gpus','Share regions across GPUs','Change region dispatch from page-assigned GPUs to all four GPUs. Each GPU can recognize four regions from the current PDF.','d8a8a05ea1b9bc9e9268','vllm_batch_4'],
  ['2414bf81cf27a32c8c2b','embed_bf16_b16','BF16 embeddings · batch 16','Change embedding inference from FP32 to BF16 and reduce the chunk batch from 32 to 16 on GPU 0.','2414bf81cf27a32c8c2b','vllm_batch_4_all_gpus'],
  ['1c2845682cf96081e45c','region_window_64','Keep 64 regions in flight','Raise the region submission window from 16 to 64 before waiting. Recognition concurrency stays at four requests per GPU.','1c2845682cf96081e45c','embed_bf16_b16'],
  ['9333355f0563a600b5d8','combined_scheduling','Assign pages dynamically','Replace fixed page assignment with a shared queue: the first free OCR worker takes the next page. Keep the 64-region window.','9333355f0563a600b5d8','region_window_64'],
  ['e618414568a8c437d78a','server_8_dynamic','Admit eight requests per GPU','Raise vLLM max-num-seqs from four to eight per GPU. Retain dynamic pages and four client requests per GPU per page worker.','9333355f0563a600b5d8','combined_scheduling'],
  ['09dd48268f7b40e92830','server_8_compiled_embeddings','Compile embedding inference','Enable torch.compile for the embedding forward with dynamic shapes. Keep BF16 precision and 16 chunks per batch.','09dd48268f7b40e92830','server_8_eager_control'],
  ['654d9b217549709275f8','server_8_ngram_gpu_4','Verify four draft tokens at once','Enable GPU n-gram speculation: propose four tokens using 2–5-token matches earlier in the same request, then verify them with the OCR model.','09dd48268f7b40e92830','server_8_compiled_embeddings'],
  ['7a9c93660221ff8d0318','available_gpu_8','Dispatch to an available GPU','Replace eager round-robin GPU assignment with a shared admission queue. Assign a region when a GPU has capacity, with eight slots per GPU.','7a9c93660221ff8d0318','eager_round_robin'],
  ['81ce12032be0a1d87ff4','decoder_tiles_fixed32','Use 32-row decoder tiles','Use fixed 32-row matrix tiles throughout the compiled BF16 decoder, with 128 columns and K=64, to reduce wasted work on small batches.','f541dac8cefb538a422c','decoder_tiles_control'],
  ['901be5b3884c7556a3dd','embedding_four_gpus','Embed across all four GPUs','Replace embedding on GPU 0 with four compiled BF16 replicas. Balance chunks across GPUs in batches of 16 and restore chunk order.','355fdb74652b8e0f18a0','embedding_single_gpu_control'],
  ['40e811453ad54f72d35a','render_in_page_workers','Render inside OCR workers','Move 150-DPI rendering from the parent process into the existing OCR page workers, preserving exact pixels and avoiding parent-to-worker image transfers.','48bcf1f74ad36b6dc4b0','render_parent_control']
];
function overviewMeasurement(id,name) {
  const r=state.runs.find(run=>run.id===id),v=r?.variants.find(variant=>variant.name===name),b=state.benchmark||{};
  const anchor=state.runs.find(run=>run.id==='d8a8a05ea1b9bc9e9268')?.variants.find(variant=>variant.name==='vllm_batch_1');
  if(!r||r.status!=='passed'||v?.status!=='passed'||!(v.baseline_qualified||v.official)||
    !anchor?.baseline_qualified||v.dataset_sha256!==anchor.dataset_sha256||
    v.metrics.total_cases!==b.pdfs||v.page_count!==b.pages||
    v.configuration?.warmup_per_case!==b.warmup||v.configuration?.repeats_per_case!==b.repeats)return null;
  return {r,v};
}
const leverColors=['#23785c','#4799aa','#6278b8','#ad8250','#87915e','#9b72b0','#bf766c','#609589','#738ca3','#b18caa','#8d806e','#6b9e65'];
function overviewContributions(key='p50_ms') {
  const throughput=key==='pages_per_minute_sequential';
  const rows=overviewMilestones.slice(1).map(([id,name,label,description,controlId,controlName],index)=>{
    const measurement=overviewMeasurement(id,name),control=overviewMeasurement(controlId,controlName);
    const after=measurement?.v.metrics[key],before=control?.v.metrics[key];
    const valid=Number.isFinite(before)&&before>0&&Number.isFinite(after)&&after>0;
    // Absolute time reductions can form shares. Speedup percentages cannot be added.
    const delta=valid?(throughput?60/before-60/after:(before-after)/1000):null;
    return {index,label,description,measurement,control,before,after,delta,
      ratio:valid?(throughput?after/before:before/after):null,color:leverColors[index]};
  });
  const positive=rows.filter(row=>row.delta>0).sort((a,b)=>b.delta-a.delta);
  const total=positive.reduce((sum,row)=>sum+row.delta,0);
  positive.forEach(row=>row.share=row.delta/total);
  return {key,throughput,rows,positive,total,regressions:rows.filter(row=>row.delta<0),missing:rows.filter(row=>row.delta===null)};
}
function leverArc(start,end) {
  const point=(radius,angle)=>[180+radius*Math.cos(angle),180+radius*Math.sin(angle)].map(n=>n.toFixed(3)).join(' ');
  end=Math.min(end,start+Math.PI*2-0.00001);
  const large=end-start>Math.PI?1:0;
  return `M ${point(143,start)} A 143 143 0 ${large} 1 ${point(143,end)} L ${point(98,end)} A 98 98 0 ${large} 0 ${point(98,start)} Z`;
}
function leverTime(value,throughput=false) {return `${num(value,3)} ${throughput?'s/page':'s'}`;}
function renderContributionChart(key='p50_ms') {
  const host=document.querySelector('#overview-contributions');
  const model=overviewContributions(key);
  let angle=-Math.PI/2;
  const slices=model.positive.map(row=>{
    const end=angle+row.share*Math.PI*2,path=leverArc(angle,end);angle=end;
    return `<path class="lever-slice" d="${path}" fill="${row.color}" data-lever="${row.index}" tabindex="0" role="button" aria-label="${esc(row.label)}: ${num(row.share*100)}% of measured time reductions" aria-describedby="lever-tooltip"></path>`;
  }).join('');
  const legendRow=(row,regression=false)=>`<li><button type="button" class="lever-legend-button ${regression?'lever-regression':''}" data-lever="${row.index}" aria-describedby="lever-tooltip"><span class="lever-dot" style="background:${regression?'#b64444':row.color}"></span><span>${esc(row.label)}</span><strong>${regression?'+'+leverTime(-row.delta,model.throughput):num(row.share*100)+'%'}</strong></button><a class="lever-run-link" href="${variantLink(row.measurement.r,row.measurement.v)}" aria-label="Open experiment: ${esc(row.label)}">↗</a></li>`;
  host.innerHTML=`<div class="lever-layout"><div class="lever-figure"><svg class="lever-donut" viewBox="0 0 360 360" role="group" aria-label="Contribution of each lever to measured time reductions"><circle cx="180" cy="180" r="121" fill="none" stroke="#edf2ef" stroke-width="45"/>${slices}<text class="lever-center-label" x="180" y="155" text-anchor="middle">MEASURED REDUCTIONS</text><text class="lever-center-value" x="180" y="193" text-anchor="middle">${leverTime(model.total,model.throughput)}</text><text class="lever-center-caption" x="180" y="219" text-anchor="middle">across ${model.positive.length} levers</text></svg><p class="lever-help">Hover, focus or tap a slice to see the tweak.</p></div><div class="lever-legend"><div class="lever-legend-heading"><span>Optimization lever</span><span>Share of reductions</span></div><ol>${model.positive.map(row=>legendRow(row)).join('')}</ol>${model.regressions.length?`<div class="lever-regressions"><p>Tradeoffs · increased time on this metric</p><ol>${model.regressions.map(row=>legendRow(row,true)).join('')}</ol></div>`:''}${model.missing.length?`<p class="tiny muted">${model.missing.length} lever(s) excluded because comparable measurements are unavailable.</p>`:''}</div></div><div class="lever-tooltip" id="lever-tooltip" role="tooltip" hidden></div><details class="lever-method"><summary>How these contributions are calculated</summary><div><p>Each slice is a lever’s positive time reduction divided by the sum of positive reductions, using its linked control and candidate. ${model.throughput?'Pages/min is converted to seconds per page (60 ÷ pages/min) before calculating reductions.':'For the selected latency metric, time reduction = control latency − candidate latency.'}</p><p>Slowdowns appear separately. These are observed effects along the tuning sequence; interactions and repeat variation mean the slices are not a causal breakdown of the final baseline-to-retained improvement. Eval speedup is shown separately in each tooltip. Repeat controls and the latest retained verification add no slices.</p></div></details>`;
  const tooltip=host.querySelector('.lever-tooltip');
  let selected=null,pinned=false;
  const clear=()=>{tooltip.hidden=true;selected=null;pinned=false;host.querySelectorAll('.is-active').forEach(el=>el.classList.remove('is-active'));};
  const position=(target,event)=>{
    const rect=target.getBoundingClientRect(),box=tooltip.getBoundingClientRect();
    const px=event?.clientX??rect.left+rect.width/2,py=event?.clientY??rect.top+rect.height/2;
    const left=Math.max(12,Math.min(window.innerWidth-box.width-12,px+18));
    const top=Math.max(12,Math.min(window.innerHeight-box.height-12,py+18));
    tooltip.style.left=left+'px';tooltip.style.top=top+'px';
  };
  const show=(target,event)=>{
    const row=model.rows[Number(target.dataset.lever)];if(!row||row.delta===null)return;
    selected=row.index;
    const format=value=>model.throughput?num(value,2)+' pages/min':duration(value);
    const gain=(row.ratio-1)*100;
    tooltip.innerHTML=`<div class="lever-tooltip-label">${row.delta>0?num(row.share*100)+'% of measured reductions':'Tradeoff · excluded from pie'}</div><strong>${esc(row.label)}</strong><p>${esc(row.description)}</p><div class="lever-tooltip-values"><span>Before<strong>${format(row.before)}</strong></span><span>After<strong>${format(row.after)}</strong></span></div><p class="lever-tooltip-effect">${leverTime(Math.abs(row.delta),model.throughput)} ${row.delta>0?'less':'more'} · ${gain>=0?'↑':'↓'} ${num(Math.abs(gain))}% eval speedup</p>`;
    tooltip.hidden=false;position(target,event);
    host.querySelectorAll('[data-lever]').forEach(el=>el.classList.toggle('is-active',Number(el.dataset.lever)===row.index));
  };
  host.onpointerover=event=>{const target=event.target.closest('[data-lever]');if(target&&!pinned)show(target,event);};
  host.onpointermove=event=>{const target=event.target.closest('[data-lever]');if(target&&!pinned&&selected!==null&&!tooltip.hidden)position(target,event);};
  host.onpointerout=event=>{if(event.target.closest('[data-lever]')&&!pinned&&!event.relatedTarget?.closest?.('[data-lever]'))clear();};
  host.leverFocusController?.abort();
  host.leverFocusController=new AbortController();
  host.addEventListener('focusin',event=>{const target=event.target.closest('[data-lever]');if(target){pinned=false;show(target);}},{signal:host.leverFocusController.signal});
  host.addEventListener('focusout',event=>{if(!pinned&&!event.relatedTarget?.closest?.('[data-lever]'))clear();},{signal:host.leverFocusController.signal});
  host.onclick=event=>{const target=event.target.closest('[data-lever]');if(!target){clear();return;}if(pinned&&selected===Number(target.dataset.lever)){clear();return;}show(target,event.detail?event:null);pinned=true;};
  host.onkeydown=event=>{if(event.key==='Escape'){clear();return;}const target=event.target.closest('path[data-lever]');if(target&&['Enter',' '].includes(event.key)){event.preventDefault();show(target);pinned=true;}};
}
function overview() {
  const b=state.benchmark||{},ocr=state.defaults.ocr;
  const baseline=overviewMeasurement('d8a8a05ea1b9bc9e9268','vllm_batch_1');
  const retained=overviewMeasurement('b75edaea6a9b5182a677','server_8_draft4_control');
  const configurations=[
    {measurement:baseline,label:'Starting point · vLLM batch 1',short:'Page-assigned OCR · FP32 embeddings on GPU 0',description:'Render pages at 150 DPI. PP-DocLayoutV3 identifies regions; PaddleOCR-VL-1.6 on vLLM recognizes one region at a time on each page’s assigned GPU. Chunk Markdown to a 512-token target, then embed with Qwen3-Embedding-0.6B in FP32, 32 chunks per batch on GPU 0.'},
    {measurement:retained,label:'Best retained configuration',short:'Shared OCR regions · compiled BF16 embeddings across four GPUs',description:'Four OCR replicas, eight requests per GPU, four draft tokens, a 64-region window, dynamic pages, available-GPU dispatch and fixed 32-row decoder tiles. Render at the same 150 DPI inside OCR workers. Keep the 512-token chunk target; compile BF16 embeddings in batches of 16 across all four GPUs.'}
  ];
  content.innerHTML=`<div class="overview">
    <header class="overview-heading"><h1>Document pipeline</h1><p class="overview-context">${esc(ocr.model.split('/').pop())} · vLLM · ${esc(b.hardware_note?.split(' · ')[0]||'4 NVIDIA L4 GPUs')} · ${num(b.pdfs,0)} frozen PDFs</p></header>
    ${pipelineDiagram()}
    <section class="card overview-comparison" aria-labelledby="overview-comparison-title">
      <div class="overview-card-top"><h2 id="overview-comparison-title">Baseline → retained configuration</h2><a href="#/doc/pages/baseline">Baseline & evaluation →</a></div>
      <div class="table-wrap"><table class="overview-comparison-table" aria-label="Baseline and retained configuration comparison"><thead><tr><th>Configuration</th><th class="number">p50 latency</th><th class="number">p95 latency</th><th class="number">Pages/min</th><th class="number">p50 vs baseline</th></tr></thead><tbody>${configurations.map((row,index)=>{
        const m=row.measurement?.v.metrics||{},speedup=row.measurement?(index?row.measurement.v.speedup:1):null;
        return `<tr class="${index?'comparison-retained':'comparison-baseline'}"><td><div class="comparison-name">${row.measurement?`<a href="${variantLink(row.measurement.r,row.measurement.v)}">${esc(row.label)} ↗</a>`:`<strong>${esc(row.label)}</strong>`}${badge(row.measurement?'passed':'pending')}</div><p class="comparison-description">${esc(row.short)}</p><details class="comparison-config"><summary>Configuration details</summary><div>${esc(row.description)}</div></details></td><td class="number"><strong>${duration(m.p50_ms)}</strong></td><td class="number"><strong>${duration(m.p95_ms)}</strong></td><td class="number"><strong>${num(m.pages_per_minute_sequential,2)}</strong></td><td class="number"><strong>${Number.isFinite(speedup)?num(speedup,2)+'×':'—'}</strong><span class="comparison-gain">${Number.isFinite(speedup)?(index?'↑ '+num((speedup-1)*100)+'%':'Baseline'):''}</span></td></tr>`;
      }).join('')}</tbody></table></div>
      <p class="overview-chart-note">${num(b.pdfs,0)} frozen PDFs · ${num(b.pages,0)} pages · Four L4 GPUs · ${num(b.warmup,0)} warm-up + ${num(b.repeats,0)} measured calls per PDF. Eval pools measured full-PDF calls; warm-ups are excluded. ${baseline&&retained?'Both configurations passed frozen checks.':'Only validated measurements are shown.'}</p>
      <div class="comparison-footer"><p>Retained p50 repeats: 2.23–2.27 s. The latest verification is shown above.</p><a class="button small" href="#/runs">All experiment runs →</a></div>
    </section>
    <section class="card overview-journey" aria-labelledby="overview-journey-title">
      <div class="overview-card-top"><div><h2 id="overview-journey-title">Where the gains came from</h2><p class="tiny muted">Share of measured time reductions. Explore a lever for the exact tweak.</p></div><label class="overview-metric-picker">Metric <select id="overview-metric" aria-label="Optimization contribution metric"><option value="p50_ms">p50 latency</option><option value="p95_ms">p95 latency</option><option value="pages_per_minute_sequential">Time per page</option></select></label></div>
      <div id="overview-contributions"></div>
    </section>
    <nav class="overview-resources" aria-label="Pipeline research and results"><a href="#/doc/pages/pipeline">Pipeline documentation →</a><a href="#/doc/experiments/028-optimization-conclusion">Sweep findings →</a><a href="#/journal">Experiment journal →</a><a href="#/datasets">Frozen PDFs →</a><a href="#/output/latest">PDF → Markdown</a><a href="#/embeddings/latest">Embeddings →</a></nav>
  </div>`;
  renderContributionChart();
  document.querySelector('#overview-metric').addEventListener('change',event=>renderContributionChart(event.target.value));
}
function runCard(r) {const b=r.variants.find(v=>v.baseline);return `<a class="run-card" href="#/run/${r.id}"><div class="run-card-top"><strong>${esc(r.name)}</strong>${badge(r.status)}</div><div class="run-meta"><span>${r.variants.length} variants</span><span>Baseline p50 ${duration(b?.metrics?.p50_ms)}</span><span>${r.variants.filter(v=>v.official).length} accepted comparisons</span></div><div class="run-path mono">${esc(r.path)}</div></a>`;}
function variantLink(r,v){return `#/run/${encodeURIComponent(r.id)}/${encodeURIComponent(v.name)}`;}
function outputLink(run,doc,page=1,mode='rendered'){
  return '#/output/'+[run.id,doc.id,page,mode].map(encodeURIComponent).join('/');
}
// Even prepared artifacts are rendered through a formatting-only allowlist.
function outputHTML(html){
  const template=document.createElement('template');template.innerHTML=html;
  const tags=new Set('P DIV SPAN H1 H2 H3 H4 H5 H6 STRONG EM B I S DEL SUB SUP UL OL LI BLOCKQUOTE PRE CODE BR HR TABLE THEAD TBODY TFOOT TR TH TD CAPTION'.split(' '));
  const copy=node=>{
    if(node.nodeType===Node.TEXT_NODE)return document.createTextNode(node.textContent);
    if(node.nodeType!==Node.ELEMENT_NODE)return document.createTextNode('');
    if(!tags.has(node.tagName))return document.createTextNode(node.textContent);
    const el=document.createElement(node.tagName.toLowerCase());
    for(const attr of ['rowspan','colspan']){
      const value=node.getAttribute(attr);
      if(['TD','TH'].includes(node.tagName)&&/^\d{1,3}$/.test(value)&&Number(value)>=1&&Number(value)<=100)el.setAttribute(attr,value);
    }
    if(node.tagName==='SPAN'&&node.className==='markdown-image-placeholder')el.className=node.className;
    el.append(...[...node.childNodes].map(copy));return el;
  };
  const fragment=document.createDocumentFragment();fragment.append(...[...template.content.childNodes].map(copy));return fragment;
}
let currentOutput=null;
async function outputView(parts,token){
  const candidates=parts[1]==='latest'||!parts[1]?state.runs:state.runs.filter(r=>r.id===parts[1]);
  if(!candidates.length){content.innerHTML=empty('No experiment run available','Run results are needed to identify the Markdown output.');return;}
  content.innerHTML=heading('Results','PDF → Markdown','Loading saved document output…');
  let run,artifact;
  for(const summary of candidates){
    run=await loadJSON(`site-data/runs/${encodeURIComponent(summary.id)}.json`);
    if(token!==generation)return;
    artifact=run.artifacts.find(a=>a.name==='output-preview.json');
    if(artifact)break;
  }
  if(!artifact){
    content.innerHTML=heading('Results','PDF → Markdown',run.name)+empty('No saved Markdown preview for this run','Open the latest run to view its prepared page outputs.','<a class="button" href="#/output/latest">Latest results →</a>');
    return;
  }
  const output=await loadJSON(artifactLink(artifact));
  if(token!==generation)return;
  if(output.kind!=='document-output-preview'||output.run_id!==run.id)throw new Error('The saved preview belongs to a different run.');
  // Preserve old bookmarks that included an experiment variant in the URL.
  const offset=run.variants.some(v=>v.name===decodeURIComponent(parts[2]||''))?3:2;
  const doc=output.documents.find(d=>d.id===decodeURIComponent(parts[offset]||''))||output.documents[0];
  const dataset=state.datasets.find(d=>d.id===output.dataset_id),pdf=dataset?.documents.find(d=>d.id===doc.id);
  if(!pdf||pdf.sha256!==doc.pdf_sha256||pdf.pages!==doc.pages.length)throw new Error('The current PDF dataset differs from this saved output.');
  const requested=Number(parts[offset+1]||1),page=Number.isInteger(requested)?Math.max(1,Math.min(doc.pages.length,requested)):1;
  const mode=parts[offset+2]==='raw'?'raw':'rendered',savedPage=doc.pages[page-1];
  const href=(p=page,m=mode)=>outputLink(run,doc,p,m);
  currentOutput={doc,page};
  content.innerHTML=`<div class="output-view">
    ${heading('Results','PDF → Markdown',`${output.documents.length} PDFs · Frozen benchmark output`,`<a class="button small" href="#/run/${run.id}">Benchmark results →</a>`)}
    <div class="output-controls">
      <label>PDF<select data-output-navigation aria-label="PDF">${output.documents.map(d=>`<option value="${outputLink(run,d,1,mode)}"${d.id===doc.id?' selected':''}>${esc(d.title)} · ${d.pages.length} pages</option>`).join('')}</select></label>
      <div class="output-page-controls"><a class="button small" href="${href(Math.max(1,page-1))}" aria-label="Previous page">←</a><select data-output-navigation aria-label="Page">${doc.pages.map(p=>`<option value="${href(p.number)}"${p.number===page?' selected':''}>Page ${p.number} of ${doc.pages.length}</option>`).join('')}</select><a class="button small" href="${href(Math.min(doc.pages.length,page+1))}" aria-label="Next page">→</a></div>
    </div>
    <div class="output-provenance">${badge('complete','Frozen reference')}<span>Every experiment that passes eval must match this Markdown exactly.</span><button class="button small" data-download-markdown>Download Markdown</button></div>
    <div class="output-panels">
      <section class="output-panel" aria-label="Source PDF"><header><h2>PDF <small>Page ${page}</small></h2><a href="${datasetFile(dataset.id,doc.id)}" download>Download PDF ↓</a></header><div class="output-pdf">${pdf.preview_pages.includes(page)?`<img src="${datasetFile(dataset.id,doc.id,page)}" alt="${esc(doc.title)}, PDF page ${page}">`:'<p>PDF page preview unavailable.</p>'}</div></section>
      <section class="output-panel" aria-label="Extracted Markdown"><header><h2>Markdown <small>Page ${page}</small></h2><nav aria-label="Markdown display"><a class="${mode==='rendered'?'active':''}" href="${href(page,'rendered')}"${mode==='rendered'?' aria-current="page"':''}>Rendered</a><a class="${mode==='raw'?'active':''}" href="${href(page,'raw')}"${mode==='raw'?' aria-current="page"':''}>Raw</a></nav></header><div class="output-markdown ${mode==='raw'?'raw':'prose'}" id="output-markdown">${mode==='raw'?`<pre><code>${esc(savedPage.markdown)}</code></pre>`:''}</div></section>
    </div>
    <p class="tiny muted output-footnote">Saved page text exactly reconstructs the frozen reference. Image placeholders indicate crops not saved with the Markdown; formulas retain their source notation.</p>
  </div>`;
  if(mode==='rendered')document.querySelector('#output-markdown').replaceChildren(outputHTML(savedPage.html));
}
let currentEmbeddings=null;
const embeddingColors=['#147d70','#5976bd','#ba712b','#9a5fa5','#bc526a','#487f91','#828841'];
const embeddingExcerpt=(point,length=115)=>point.text.replace(/<[^>]*>/g,' ').replace(/\s+/g,' ').trim().slice(0,length);
function embeddingVisible(){
  const e=currentEmbeddings;
  return e.data.points.filter(p=>(!e.document||p.document_id===e.document)&&(!e.search||p.text.toLowerCase().includes(e.search))).map(p=>p.index);
}
function embeddingNeighbors(){
  const e=currentEmbeddings,selected=e.data.points[e.selected];
  if(!selected)return [];
  return e.data.similarities[e.selected].map((score,index)=>({index,score}))
    .filter(n=>n.index!==e.selected&&(!e.otherDocuments||e.data.points[n.index].document_id!==selected.document_id))
    .sort((a,b)=>b.score-a.score||a.index-b.index).slice(0,5);
}
function embeddingMap(visible,neighbors){
  const e=currentEmbeddings,allowed=new Set(visible),near=new Set(neighbors.map(n=>n.index));
  const selected=e.positions[e.selected],width=680/e.zoom,height=460/e.zoom;
  const cx=e.zoom>1&&selected?selected[0]:340,cy=e.zoom>1&&selected?selected[1]:230;
  const map=document.querySelector('#embedding-map');
  map.setAttribute('viewBox',`${cx-width/2} ${cy-height/2} ${width} ${height}`);
  const lines=selected?neighbors.filter(n=>allowed.has(n.index)).map(n=>`<line x1="${selected[0]}" y1="${selected[1]}" x2="${e.positions[n.index][0]}" y2="${e.positions[n.index][1]}" class="embedding-connection"/>`).join(''):'';
  // Draw the selected point last so coincident points do not cover its highlight.
  const ordered=[...e.data.points.filter(p=>p.index!==e.selected),...(e.selected>=0?[e.data.points[e.selected]]:[])];
  map.innerHTML=`<title>Chunk map</title><desc>Two-dimensional PCA projection. Click or focus a chunk and press Enter to read it. Similarity rankings use the original vectors.</desc>
    <line x1="30" y1="424" x2="654" y2="424" class="embedding-axis"/><line x1="30" y1="26" x2="30" y2="424" class="embedding-axis"/>
    <text x="340" y="448" class="embedding-axis-label">PC1 · ${num(e.data.projection.explained_variance_ratio[0]*100)}% variance</text>
    <text x="17" y="230" transform="rotate(-90 17 230)" class="embedding-axis-label">PC2 · ${num(e.data.projection.explained_variance_ratio[1]*100)}%</text>${lines}
    ${ordered.map(p=>{
      const xy=e.positions[p.index],doc=e.documents.get(p.document_id),active=p.index===e.selected,neighbor=near.has(p.index),shown=allowed.has(p.index);
      return `<g class="embedding-node ${active?'selected':''} ${neighbor?'neighbor':''} ${shown?'':'dimmed'}" data-embedding-select="${p.index}" role="button" tabindex="${shown?'0':'-1'}" aria-label="${esc(doc.title)}, ${esc(p.chunk_id)}${active?', selected':''}" aria-pressed="${active}"${shown?'':' aria-hidden="true"'}><title>${esc(doc.title)} · ${esc(p.chunk_id)}\n${esc(embeddingExcerpt(p))}</title><circle class="embedding-hit" cx="${xy[0]}" cy="${xy[1]}" r="11"/>${active||neighbor?`<circle class="embedding-ring" cx="${xy[0]}" cy="${xy[1]}" r="${active?10:8.5}"/>`:''}<circle class="embedding-dot" cx="${xy[0]}" cy="${xy[1]}" r="${active?6.5:5}" fill="${doc.color}"/></g>`;
    }).join('')}`;
  document.querySelector('#embedding-count').textContent=`${visible.length} / ${e.data.points.length} chunks`;
  document.querySelector('#embedding-no-match').hidden=visible.length>0;
  document.querySelectorAll('[data-embedding-document]').forEach(button=>button.setAttribute('aria-pressed',String(button.dataset.embeddingDocument===e.document)));
}
function embeddingInspector(neighbors){
  const e=currentEmbeddings,target=document.querySelector('#embedding-inspector'),point=e.data.points[e.selected];
  if(!point){target.innerHTML='<p class="muted">No chunks match these filters.</p>';return;}
  const doc=e.documents.get(point.document_id);
  target.innerHTML=`<div class="embedding-selected">
    <label class="embedding-select-label" for="embedding-chunk">Selected chunk</label>
    <select id="embedding-chunk" aria-label="Selected chunk">${e.data.documents.map(d=>`<optgroup label="${esc(d.title)}">${e.data.points.filter(p=>p.document_id===d.id).map(p=>`<option value="${p.index}"${p.index===point.index?' selected':''}>${esc(p.chunk_id)} · ${esc(embeddingExcerpt(p,60))}</option>`).join('')}</optgroup>`).join('')}</select>
    <div class="embedding-chunk-meta"><span class="embedding-doc-dot" style="background:${doc.color}"></span><strong>${esc(doc.title)}</strong><span>${num(point.tokens,0)} tokens</span></div>
    <div class="embedding-text prose" id="embedding-selected-text"></div>
    <a class="tiny" href="${outputLink(e.run,{id:point.document_id})}">Open PDF → Markdown ↗</a>
  </div>
  <div class="embedding-neighbor-heading"><h2>Closest passages</h2><label><input id="embedding-other-pdfs" type="checkbox"${e.otherDocuments?' checked':''}> Other PDFs only</label></div>
  <p class="embedding-neighbor-note">Cosine similarity in ${num(e.data.dimensions,0)} dimensions · self excluded</p>
  <div class="embedding-neighbors">${neighbors.map(n=>{
    const p=e.data.points[n.index],d=e.documents.get(p.document_id);
    return `<button class="embedding-neighbor" data-embedding-select="${p.index}" aria-label="Explore ${esc(d.title)}, ${esc(p.chunk_id)}"><div class="embedding-neighbor-top"><span><span class="embedding-doc-dot" style="background:${d.color}"></span>${esc(d.title)}</span><strong>${num(n.score,3)}</strong></div><small>${esc(p.chunk_id)}</small><p>${esc(embeddingExcerpt(p,145))}${p.text.length>145?'…':''}</p></button>`;
  }).join('')||'<p class="muted">No other passages available.</p>'}</div>`;
  document.querySelector('#embedding-selected-text').replaceChildren(outputHTML(point.html));
}
function updateEmbeddings(){
  if(!currentEmbeddings||!document.querySelector('#embedding-map'))return;
  const visible=embeddingVisible();
  if(!visible.includes(currentEmbeddings.selected))currentEmbeddings.selected=visible[0]??-1;
  const neighbors=embeddingNeighbors();embeddingMap(visible,neighbors);embeddingInspector(neighbors);
}
function selectEmbedding(index){
  const e=currentEmbeddings;
  if(!e||!Number.isInteger(index)||!e.data.points[index])return;
  if(!embeddingVisible().includes(index)){
    e.document='';e.search='';document.querySelector('#embedding-text-filter').value='';
  }
  e.selected=index;updateEmbeddings();
}
async function embeddingsView(parts,token){
  const candidates=parts[1]==='latest'||!parts[1]?state.runs:state.runs.filter(r=>r.id===parts[1]);
  if(!candidates.length){content.innerHTML=empty('No saved embeddings yet','A frozen reference set is needed for this view.');return;}
  content.innerHTML=heading('Results','Embeddings','Loading frozen chunk embeddings…');
  // A new optimization run does not replace the frozen reference visualization.
  let run,artifact;
  for(const summary of candidates){
    run=await loadJSON(`site-data/runs/${encodeURIComponent(summary.id)}.json`);
    if(token!==generation)return;
    artifact=run.artifacts.find(a=>a.name==='embedding-preview.json');
    if(artifact)break;
  }
  if(!artifact){content.innerHTML=heading('Results','Embeddings',run.name)+empty('No chunk map prepared for this run','The map uses saved embeddings and reconstructed chunk text.');return;}
  const data=await loadJSON(artifactLink(artifact));
  if(token!==generation)return;
  const reference=run.variants.find(v=>v.name===data.reference_variant);
  if(data.kind!=='document-embedding-preview'||data.run_id!==run.id||reference?.dataset_sha256!==data.dataset_sha256)throw new Error('The embedding map differs from this run’s frozen references.');
  const dataset=state.datasets.find(d=>d.id===data.dataset_id);
  if(data.documents.some(d=>dataset?.documents.find(pdf=>pdf.id===d.id)?.sha256!==d.pdf_sha256))throw new Error('The current PDF dataset differs from these saved embeddings.');
  if(!data.points.length||data.points.some((p,i)=>p.index!==i||p.position.length!==2||!p.position.every(Number.isFinite))||data.similarities.length!==data.points.length||data.similarities.some(row=>row.length!==data.points.length||!row.every(Number.isFinite)))throw new Error('The embedding map is incomplete.');
  const xs=data.points.map(p=>p.position[0]),ys=data.points.map(p=>p.position[1]);
  const xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys);
  const scale=Math.min(580/Math.max(xmax-xmin,1e-9),350/Math.max(ymax-ymin,1e-9));
  currentEmbeddings={
    run,data,documents:new Map(data.documents.map((d,i)=>[d.id,{...d,color:embeddingColors[i%embeddingColors.length]}])),
    positions:data.points.map(p=>[342+(p.position[0]-(xmin+xmax)/2)*scale,225-(p.position[1]-(ymin+ymax)/2)*scale]),
    selected:data.points.find(p=>p.tokens>=100&&!p.text.trim().startsWith('<'))?.index||0,
    document:'',search:'',otherDocuments:false,zoom:1,
  };
  const explained=data.projection.explained_variance_ratio.reduce((sum,v)=>sum+v,0);
  content.innerHTML=`<div class="embedding-view">
    ${heading('Results','Embeddings','Click a chunk to read it and explore similar passages.',badge('complete','Frozen references'))}
    <p class="embedding-context">${num(data.points.length,0)} chunks · ${data.documents.length} PDFs · ${num(data.dimensions,0)} dimensions · ${esc(data.model)}</p>
    <div class="embedding-layout">
      <section class="card embedding-map-card" aria-labelledby="embedding-map-title">
        <div class="embedding-map-heading"><h2 id="embedding-map-title">Chunk map</h2><span id="embedding-count"></span></div>
        <div class="embedding-map-tools"><input id="embedding-text-filter" type="search" placeholder="Filter chunk text…" aria-label="Filter chunk text"><div role="group" aria-label="Map zoom"><button class="button small" data-embedding-zoom="out" aria-label="Zoom out">−</button><button class="button small" data-embedding-zoom="in" aria-label="Zoom in">+</button><button class="button small" data-embedding-zoom="reset">Reset view</button></div></div>
        <div class="embedding-plot"><svg id="embedding-map" viewBox="0 0 680 460" role="group" aria-label="Interactive chunk map"></svg><p id="embedding-no-match" hidden>No chunks match these filters.</p></div>
        <div class="embedding-legend" aria-label="Filter by PDF"><button data-embedding-document="" aria-pressed="true">All PDFs</button>${data.documents.map(d=>`<button data-embedding-document="${esc(d.id)}" aria-pressed="false"><span class="embedding-doc-dot" style="background:${currentEmbeddings.documents.get(d.id).color}"></span>${esc(d.title)} <small>${d.chunks}</small></button>`).join('')}</div>
        <p class="embedding-map-note">2D PCA retains ${num(explained*100)}% of vector variance. Map distances are approximate; neighbor scores use all ${num(data.dimensions,0)} dimensions.</p>
      </section>
      <aside class="card embedding-inspector" id="embedding-inspector" aria-label="Selected chunk and nearest neighbors"></aside>
    </div>
  </div>`;
  updateEmbeddings();
}
function experimentNote(r,v=null){
  const status=['superseded','invalidated'].includes(r.status)?r.status:v?.status||r.status;
  if(['failed','superseded','invalidated'].includes(status)){
    const note=r.baseline_note;if(!note)return '';
    const original=state.runs.find(x=>x.path===note.original_run);
    return `<div class="callout warning"><strong>${esc(note.title)}</strong> ${esc(note.details||note.summary)}${original?` <a href="#/run/${original.id}">Original PaddlePaddle baseline →</a>`:''}</div>`;
  }
  const selected=v?[v]:r.variants;
  return selected.filter(item=>item.status==='passed'&&item.outcome_note?.summary).map(item=>{
    const note=item.outcome_note,profiles=profilesForRun(r,item);
    const evidence=profiles.filter(p=>p.status==='complete').slice(0,2);
    return `<div class="callout experiment-outcome"><strong>${esc(note.summary)}</strong>${note.next?`<p><strong>Next:</strong> ${esc(note.next)}</p>`:''}${evidence.length?`<p class="tiny">Profiling: ${evidence.map(p=>`<a href="#/profile/${esc(p.id)}">${esc((r.profile_links||[]).find(link=>link.path===p.path&&link.variant===item.name)?.label||p.pdf?.name||'Study')} ↗</a>`).join(' · ')}</p>`:''}</div>`;
  }).join('');
}
function profileRun(p){return state.runs.find(r=>(r.profile_links||[]).some(link=>link.path===p.path))||state.runs.find(r=>r.path===p.baseline_run);}
function profileName(p){const run=profileRun(p),link=run?.profile_links?.find(item=>item.path===p.path);return link?.label||run?.variants.find(v=>v.name===p.variant)?.display_name||run?.name||p.label||'Profile';}
function observedCaseMs(c){
  if(c?.p50_ms!=null)return c.p50_ms;
  const values=(c?.runs||[]).filter(r=>r.phase==='measured'&&Number.isFinite(r.latency_ms)).map(r=>r.latency_ms).sort((a,b)=>a-b);
  if(!values.length)return null;
  const middle=Math.floor(values.length/2);
  return values.length%2?values[middle]:(values[middle-1]+values[middle])/2;
}
function profilesForRun(r,v){return state.profiles.filter(p=>(p.baseline_run===r.path&&(!v||!p.variant||p.variant===v.name))||(r.profile_links||[]).some(link=>link.path===p.path&&(!v||link.variant===v.name))).sort((a,b)=>String(b.created_at||b.path).localeCompare(String(a.created_at||a.path)));}
function profileContext(r,v,p){const link=(r.profile_links||[]).find(item=>item.path===p.path&&item.variant===v.name);return {name:link?.label||v.display_name||r.name,reused:p.baseline_run!==r.path||Boolean(p.variant&&p.variant!==v.name)};}
function profileCard(p,context=null){return `<a class="run-card" href="#/profile/${p.id}"><div class="run-card-top"><strong>${esc(context?.name||profileName(p))} · ${esc(p.pdf?.name)}</strong>${badge(p.status,p.status==='complete'?'Profile captured':statusLabel(p.status))}</div>${context?.reused?'<p class="tiny muted">Retained earlier diagnostic; its capture configuration is recorded on the profile page. Not part of this evaluation’s latency samples.</p>':''}${p.findings?`<p>${esc(p.findings.summary)}</p><p class="tiny"><strong>Next:</strong> ${esc(p.findings.next_step)}</p>`:''}<div class="run-meta"><span>${duration(p.summary?.median_total_ms)} median call</span><span>${esc(p.summary?.dominant_stage||'No stage data')} ${p.summary?.dominant_stage?'dominates':''}</span><span>Diagnostic</span><span>${esc(date(p.created_at))}</span></div></a>`;}
function improvement(v){
  if(v.official&&Number.isFinite(v.speedup)){
    const percent=(v.speedup-1)*100;
    return `<strong class="speedup-value ${percent<0?'regression':''}">${percent>=0?'+':''}${num(percent)}%</strong><span class="speedup-note">${num(v.speedup,2)}× · eval accepted</span>`;
  }
  if(v.baseline&&v.baseline_qualified)return '<strong class="baseline-value">Baseline</strong><span class="speedup-note">Comparison control</span>';
  return `<span class="muted" title="${esc(v.comparison_note||'No accepted comparison')}">—</span>`;
}
function metricChange(v,key){
  const ratio=v.official?v.comparison_metrics?.[key]?.speedup:null;
  if(!Number.isFinite(ratio))return '';
  const percent=(ratio-1)*100;
  const direction=percent>0?'↑':percent<0?'↓':'→';
  const label=key==='pages_per_minute_sequential'?'Pages/min':'Latency';
  return `<span class="metric-change ${percent>0?'gain':percent<0?'loss':'neutral'}" title="${label} speedup vs baseline: (eval speedup − 1) × 100. Positive means faster.">${direction} ${num(Math.abs(percent))}%</span>`;
}
function previousSourceLink(source,label) {
  const r=state.runs.find(item=>item.id===source.run_id);
  const v=r?.variants.find(item=>item.name===source.variant);
  const href=v?variantLink(r,v):artifactLink({path:source.report_path});
  return `<a href="${href}" title="${esc(v?runDisplayName(r,v):source.variant)}">${esc(label)} ↗</a>`;
}
function previousImprovement(v) {
  if(!v.official)return '<span class="muted">—</span>';
  const previous=v.previous_comparison;
  if(previous?.kind==='repeat')return '<span class="previous-repeat">Repeat</span><span class="speedup-note">Same configuration</span>';
  const ratio=previous?.metrics?.p50_ms?.speedup;
  if(!Number.isFinite(ratio))return `<span class="muted" title="${esc(previous?.error||'No recorded previous-best comparison')}">—</span>`;
  const percent=(ratio-1)*100;
  return `<strong class="previous-value ${percent<0?'regression':''}">${percent>0?'↑':percent<0?'↓':'→'} ${num(Math.abs(percent))}%</strong><span class="speedup-note">${previousSourceLink(previous.source,previous.source.fresh_control?'Fresh control':'Previous best')}</span>`;
}
function previousComparisonDetails(v) {
  const previous=v.official?v.previous_comparison:null;
  if(!previous?.metrics)return '';
  return `<section class="section previous-comparison">${sectionHead('Improvement vs previous best','Uses the prior configuration tested against this experiment, with its fresh control where recorded.')}<p class="tiny">${previousSourceLink(previous.source,'Comparison measurement')} · <a href="${artifactLink({path:previous.source.report_path})}">Exact report ↗</a></p><div class="table-wrap"><table><thead><tr><th>Metric</th><th class="number">Previous best / control</th><th class="number">This experiment</th><th class="number">Improvement</th></tr></thead><tbody>${[['p50_ms','p50 latency'],['p95_ms','p95 latency'],['pages_per_minute_sequential','Pages/min']].map(([key,label])=>{
    const m=previous.metrics[key],percent=(m.speedup-1)*100;
    const format=value=>key==='pages_per_minute_sequential'?num(value,2):duration(value);
    return `<tr><td>${label}</td><td class="number">${format(m.baseline)}</td><td class="number">${format(m.optimized)}</td><td class="number"><span class="metric-change ${percent<0?'loss':'gain'}">${percent<0?'↓':'↑'} ${num(Math.abs(percent))}%</span></td></tr>`;
  }).join('')}</tbody></table></div></section>`;
}
function evaluationSummary(variants){
  return `<section class="section">${sectionHead('Eval metrics','p50 and p95 pool all measured full-PDF calls. Warm-ups are excluded. Pages/min uses total measured pages divided by total measured time.')}<div class="table-wrap"><table><thead><tr><th>Experiment</th><th class="number">p50</th><th class="number">p95</th><th class="number">Pages/min</th><th class="number">p50 improvement</th><th class="number">p95 speedup</th><th class="number">Pages/min speedup</th><th class="number">Suite runtime</th></tr></thead><tbody>${variants.map(v=>`<tr><td>${esc(v.presentation?.title||v.display_name||v.name)}</td><td class="number">${duration(v.metrics.p50_ms)}${metricChange(v,'p50_ms')}</td><td class="number">${duration(v.metrics.p95_ms)}${metricChange(v,'p95_ms')}</td><td class="number">${num(v.metrics.pages_per_minute_sequential,2)}${metricChange(v,'pages_per_minute_sequential')}</td><td class="number">${improvement(v)}</td><td class="number">${v.official?num(v.comparison_metrics.p95_ms.speedup,2)+'×':'—'}</td><td class="number">${v.official?num(v.comparison_metrics.pages_per_minute_sequential.speedup,2)+'×':'—'}</td><td class="number">${suiteRuntime(v)}</td></tr>`).join('')}</tbody></table></div></section>`;
}
function linkedRunVariants(links) {
  return (links||[]).flatMap(link=>{
    const r=state.runs.find(item=>item.path===link.path);
    const v=r?.variants.find(item=>item.name===link.variant);
    return v?[{r,v}]:[];
  });
}
function isGroupedControl(v) {
  // Keep a control visible if its candidate is unavailable; never hide the baseline.
  return !v.baseline&&linkedRunVariants(v.control_for).length>0;
}
function runDisplayName(r,v) {return v.presentation?.title||v.display_name||r.name;}
function bestRunEntry(entries) {
  return entries.filter(({r,v})=>r.status==='passed'&&v.status==='passed'&&v.official&&Number.isFinite(v.speedup))
    .reduce((best,item)=>!best||item.v.speedup>best.v.speedup?item:best,null);
}
function supportingExperiments(r,v) {
  const related=linkedRunVariants(v.presentation?.supporting_experiments);
  if(!related.length)return '';
  return `<section class="section supporting-experiments">${sectionHead('Slower follow-up','Retained here as evidence for keeping eight requests per GPU.')}<div class="table-wrap"><table><thead><tr><th>Experiment</th><th class="number">p50</th><th class="number">p95</th><th class="number">Pages/min</th></tr></thead><tbody>${related.map(({r,v})=>`<tr><td><a href="${variantLink(r,v)}">${esc(runDisplayName(r,v))} ↗</a></td><td class="number">${duration(v.metrics.p50_ms)}</td><td class="number">${duration(v.metrics.p95_ms)}</td><td class="number">${num(v.metrics.pages_per_minute_sequential,2)}</td></tr>`).join('')}</tbody></table></div><p class="tiny muted">Full measurements and profiling remain available through the experiment link.</p></section>`;
}
function controlEvidence(r,v) {
  const controls=linkedRunVariants(v.controls);
  const candidates=linkedRunVariants(v.control_for);
  if(!controls.length&&!candidates.length)return '';
  if(candidates.length)return `<div class="callout control-context"><strong>Repeat control.</strong> This reruns the previous configuration to check timing variation. Related experiments: ${candidates.map(({r,v})=>`<a href="${variantLink(r,v)}">${esc(v.display_name||v.name)} →</a>`).join(' · ')}</div>`;
  return `<section class="section control-evidence">${sectionHead('Repeat control','Previous configuration remeasured for this comparison. Headline improvement still uses the original batch 1 baseline.')}<div class="table-wrap"><table><thead><tr><th>Configuration</th><th>Status</th><th class="number">p50</th><th class="number">p95</th><th class="number">Pages/min</th><th class="number">Suite runtime</th></tr></thead><tbody>${controls.map(({r,v})=>`<tr><td><a href="${variantLink(r,v)}">${esc(v.display_name||v.name)} ↗</a><div class="tiny muted">${esc(r.timestamp)}</div></td><td>${badge(['superseded','invalidated'].includes(r.status)?r.status:v.status)}</td><td class="number">${duration(v.metrics.p50_ms)}</td><td class="number">${duration(v.metrics.p95_ms)}</td><td class="number">${num(v.metrics.pages_per_minute_sequential,2)}</td><td class="number">${suiteRuntime(v)}</td></tr>`).join('')}</tbody></table></div><p class="tiny muted">Open the control for its per-PDF measurements, profiles and original artifacts.</p></section>`;
}
function runExpansionContent(r,v) {
  const profiles=profilesForRun(r,v);
  return `<div class="run-expanded-content">${experimentNote(r,v)}${r.invalidation?`<div class="callout warning"><strong>OCR quality invalidated.</strong> ${esc(r.invalidation.reason)}</div>`:''}${v.comparison_note?`<p class="tiny muted">Comparison unavailable: ${esc(v.comparison_note)}</p>`:''}${evaluationSummary([v])}${previousComparisonDetails(v)}${supportingExperiments(r,v)}${v.presentation?.supporting_experiments?.length?'':controlEvidence(r,v)}${perDocumentResults(v)||'<p class="muted">Per-PDF results will appear when evaluation finishes.</p>'}${v.timing?.stages?.length?`<section class="section">${sectionHead('Median stage timings','Percentage shares use mean times; warm-ups excluded.')}<div class="card">${charts(v.timing)}</div></section>`:''}<section class="section">${sectionHead('Profiling results','Diagnostic evidence · newest first')}<div class="run-list">${profiles.length?profiles.map(p=>profileCard(p,profileContext(r,v,p))).join(''):'<p class="muted">No profile linked to this configuration.</p>'}</div></section><a class="button small" href="${variantLink(r,v)}">Full experiment details ↗</a></div>`;
}
let runDetailRequests=new Map();
async function loadRunExpansion(panel) {
  if(!panel||panel.dataset.loaded==='true'||panel.dataset.loading==='true')return;
  const token=generation,id=panel.dataset.runId;
  panel.dataset.loading='true';
  panel.setAttribute('aria-busy','true');
  panel.firstElementChild.innerHTML='<div class="run-expanded-content" role="status">Loading PDF timings and profiling details…</div>';
  try {
    let request=runDetailRequests.get(id);
    if(!request){
      const requests=runDetailRequests;
      request=loadJSON(`site-data/runs/${encodeURIComponent(id)}.json`);
      requests.set(id,request);
      request.catch(()=>{if(requests.get(id)===request)requests.delete(id);});
    }
    const r=await request;
    if(token!==generation||!panel.isConnected)return;
    const v=r.variants.find(item=>item.name===panel.dataset.variant);
    if(!v)throw new Error('Variant not found');
    panel.firstElementChild.innerHTML=runExpansionContent(r,v);
    panel.dataset.loaded='true';
  }catch(error){
    if(token===generation&&panel.isConnected){
      runDetailRequests.delete(id);
      panel.firstElementChild.innerHTML=`<div class="run-expanded-content"><p class="error-list">Could not load details: ${esc(error.message)}</p><button class="button small" data-run-retry="${esc(panel.id)}">Retry details</button></div>`;
    }
  }finally{
    delete panel.dataset.loading;
    panel.removeAttribute('aria-busy');
  }
}
function runsSummaryTable(list) {
  const entries=list.flatMap(r=>[...r.variants].reverse().map(v=>({r,v})));
  const best=bestRunEntry(entries);
  const visible=entries.filter(item=>item===best||(!isGroupedControl(item.v)&&!linkedRunVariants(item.v.presentation?.supporting_for).length));
  const ordered=best?[best,...visible.filter(item=>item!==best)]:visible;
  const rows=ordered.map(({r,v})=>{
    const isBest=r===best?.r&&v===best?.v;
    const activeVariant=!r.lifecycle?.variant||r.lifecycle.variant===v.name;
    const rowStatus=['invalidated','superseded'].includes(r.status)?r.status:v.status!=='pending'?v.status:r.lifecycle?.phase==='evaluation'&&!activeVariant?'pending':r.status;
    const phase=r.lifecycle?.phase&&r.lifecycle.phase!=='finished'&&activeVariant?`<div class="tiny muted">${esc(r.lifecycle.phase)}${r.progress?` · ${r.progress.completed_calls} calls / ${r.progress.completed_pages} pages`:''}</div>`:'';
    const key=`run-details-${r.id}-${v.name}`;
    const count=v.metrics.total_cases??v.cases?.length;
    const cohort=count?`${count} PDFs${v.page_count?' · '+v.page_count+' pages':''} · ${v.configuration?.repeats_per_case??r.experiment?.repeats??'—'} measured/PDF`:'Evaluation pending';
    const expanded=`<tr class="run-expansion run-status-${esc(rowStatus)}" id="${esc(key)}" data-run-id="${esc(r.id)}" data-variant="${esc(v.name)}" hidden><td colspan="8"></td></tr>`;
    const description=v.presentation?.description||v.change_description;
    return `<tr class="run-summary-row run-status-${esc(rowStatus)} ${isBest?'run-best':''} ${r.status==='invalidated'?'invalidated-run':''}"><td class="summary-run"><div class="run-name-line"><button class="run-toggle" data-run-toggle="${esc(key)}" aria-expanded="false" aria-controls="${esc(key)}" aria-label="Expand ${esc(runDisplayName(r,v))}">▸</button><div>${isBest?'<div class="best-run-label">★ Best measured p50</div>':''}<a href="${variantLink(r,v)}"><strong>${esc(runDisplayName(r,v))}</strong></a>${description?`<div class="tiny muted">${esc(description)}</div>`:''}<div class="run-cohort">${esc(cohort)} · ${esc(r.timestamp)}</div>${r.summary_note?`<div class="tiny muted">${esc(r.summary_note)}</div>`:''}</div></div></td><td class="number improvement-cell">${improvement(v)}</td><td class="number previous-cell">${previousImprovement(v)}</td><td>${badge(rowStatus)}${phase}</td><td class="number">${duration(v.metrics.p50_ms)}${metricChange(v,'p50_ms')}</td><td class="number">${duration(v.metrics.p95_ms)}${metricChange(v,'p95_ms')}</td><td class="number">${num(v.metrics.pages_per_minute_sequential,2)}${metricChange(v,'pages_per_minute_sequential')}</td><td class="number">${suiteRuntime(v)}</td></tr>${expanded}`;
  }).join('');
  return `<div class="table-wrap"><table class="runs-summary" aria-label="Experiment run summary"><thead><tr><th scope="col">Experiment</th><th scope="col" class="number">p50 vs baseline</th><th scope="col" class="number">p50 vs previous best</th><th scope="col">Status</th><th scope="col" class="number">p50</th><th scope="col" class="number">p95</th><th scope="col" class="number">Pages/min</th><th scope="col" class="number" title="Full evaluation wall time, including setup, warm-ups and validation">Suite runtime</th></tr></thead><tbody>${rows}</tbody></table></div><p class="tiny muted comparison-explainer">p50 improvement = (baseline p50 / experiment p50 − 1) × 100. +100% means 2× speedup. Percentages appear only for comparisons accepted by eval on matching PDFs, references, hardware and settings. Superseded runs retain their raw metrics.</p><p class="tiny muted">Previous best uses the prior configuration each experiment built on, with its recorded fresh control where available. The link identifies the exact comparison. Repeats are labeled separately. Small indicators under p50, p95 and pages/min compare with the original baseline; expand for all three changes vs previous best.</p><p class="tiny muted">Suite runtime includes evaluator worker setup, warm-ups, measured calls, validation and comparison. Inference-server startup, environment inventory, reference preparation and profiling are separate.</p>`;
}
function runs() {
  // Details are shared between variants only for this visit; Refresh reads fresh reports.
  runDetailRequests=new Map();
  const list=state.runs;
  const b=state.benchmark||{};
  const method=`Current benchmark: ${b.pdfs||'—'} frozen PDFs · ${b.pages||'—'} pages. Each PDF: ${b.warmup||1} warm-up excluded + ${b.repeats||5} measured calls. PDFs run sequentially.`;
  content.innerHTML=heading('Measurements','Experiment runs','Best measured p50 pinned first; remaining experiments newest first. Expand for controls, document timings and profiles.')+`<div class="benchmark-note"><strong>Fixed hardware</strong> ${esc(b.hardware_note||'4 NVIDIA L4 GPUs')}<p>${esc(method)}</p><p>Comparison baseline: vLLM batch 1. Batch invariance is enabled for current runs.</p></div>`+(list.length?`<div id="run-summary">${runsSummaryTable(list)}</div>`:`<div class="card">${empty('No matching runs yet','PDF sweep reports appear here automatically.','<a class="button" href="#/doc/pages/baseline">Set up the baseline →</a>')}</div>`);
}
function journal() {const docs=state.documents.filter(d=>d.kind==='experiment');content.innerHTML=heading('Research notes','Experiment journal','The reasoning behind each run: evidence, hypothesis, change, result, and decision.')+`<div class="callout">Start with an observation from a profile. Candidate experiments are a blueprint, not a fixed execution order.</div><div class="run-list">${docs.map(d=>`<a class="run-card" href="#/doc/${d.id}"><div class="run-card-top"><strong>${esc(d.title)}</strong>${badge(d.status||'planned')}</div><p class="tiny muted">${esc(d.description)}</p><div class="run-meta"><span>${esc(d.date)}</span><span>${d.run?'Run linked':'No run linked'}</span><span>${d.profile?'Profile linked':'No profile linked'}</span></div></a>`).join('')}</div><section class="section"><h2>Create a journal entry</h2><pre><code>.venv/bin/python -m lab.note experiment-name --title "The question we are testing"</code></pre><p class="tiny muted">Edit the generated Markdown in docs/experiments/. Add run and profile paths to connect the evidence.</p></section>`;}
function charts(summary) {if(!summary?.stages?.length)return '<p class="muted tiny">No complete stage timings are available.</p>';return summary.stages.map(s=>`<div class="stage-row"><span>${esc(s.name)}</span><progress max="1" value="${Math.max(0,Math.min(1,s.share))}" aria-label="${esc(s.name)} share of total time"></progress><span class="number">${duration(s.median_ms)}</span><span class="number muted">${num(s.share*100,0)}%</span></div>`).join('')+`<p class="tiny muted">Median stage durations · percentage shares use mean stage time / mean total call time · ${summary.calls} timed calls · warm-ups excluded</p>`;}
function fileSize(bytes) {const [divisor,unit]=bytes>=1e9?[1e9,'GB']:bytes>=1e6?[1e6,'MB']:[1e3,'KB'];return `${num(bytes/divisor,2)} ${unit}`;}
function artifacts(items) {return `<section class="section">${sectionHead('Artifacts','Download saved measurements and profiling summaries.')}<div class="artifact-grid">${items.map(a=>`<a class="artifact" href="${artifactLink(a)}" download><span class="artifact-name mono">${esc(a.name)}</span><small>${fileSize(a.bytes)} ↓</small></a>`).join('')}</div></section>`;}
function notesFor(kind,path) {const docs=state.documents.filter(d=>d.kind==='experiment'&&d[kind]===path);return `<section class="section">${sectionHead('Linked journal notes')}${docs.length?docs.map(d=>`<a class="resource" href="#/doc/${d.id}">${esc(d.title)} →</a>`).join(''):'<p class="tiny muted">No interpretation recorded yet. Link this artifact directory in a journal page’s metadata.</p>'}</section>`;}
function perDocumentResults(variant) {
  if(!variant.cases?.length)return '';
  const repeated=variant.cases.some(c=>c.runs.filter(r=>r.phase==='measured').length>1);
  return `<section class="section">${sectionHead('Per-PDF results',variant.baseline?(variant.status==='passed'?'Baseline · compare each document with itself in later experiments.':'Comparison control · '+variant.name):variant.name)}<div class="table-wrap"><table><thead><tr><th>PDF</th><th>Pages</th><th>Warm-up</th><th>${repeated?'Median timed call':'Timed call'}</th>${repeated?'<th>p95</th>':''}<th>Timed calls</th><th>Reference match</th><th>Min cosine</th></tr></thead><tbody>${variant.cases.map(c=>{
    const dataset=state.datasets.find(d=>d.documents.some(doc=>doc.id===c.id));
    const doc=dataset?.documents.find(doc=>doc.id===c.id);
    const measured=c.runs.filter(r=>r.phase==='measured'),warm=c.runs.filter(r=>r.phase==='warmup');
    const label=doc?`<a href="#/dataset/${encodeURIComponent(dataset.id)}/${encodeURIComponent(doc.id)}/1">${esc(doc.title)} ↗</a>`:esc(c.id);
    return `<tr><td>${label}${c.failures?.length&&!variant.comparison_context?`<p class="error-list">${c.failures.map(esc).join('<br>')}</p>`:''}</td><td>${doc?.pages??'—'}</td><td>${warm.map(r=>duration(r.latency_ms)).join(', ')||'—'}</td><td class="number"><strong>${duration(observedCaseMs(c))}</strong></td>${repeated?`<td>${duration(c.p95_ms)}</td>`:''}<td>${measured.length}</td><td>${variant.comparison_context?badge('completed','Engine changed'):variant.verification_context&&!c.pass?badge('checking','Varies'):badge(c.pass?'passed':'failed')}</td><td>${num(c.min_cosine,5)}</td></tr>`;
  }).join('')}</tbody></table></div><p class="tiny muted">Durations are in seconds. Warm-ups are excluded from measured latency. ${repeated?'Inspect the individual repeats in the report.':'One timed call per PDF gives a latency observation, not a repeat distribution.'}</p></section>`;
}
function runView(r,variantName=null) {
  const variants=variantName?r.variants.filter(v=>v.name===variantName):r.variants;
  if(!variants.length)throw new Error('Variant not found');
  const selected=variantName?variants[0]:null;
  const active=r.variants.find(v=>v.name===r.lifecycle?.variant);
  const viewStatus=!selected||['superseded','invalidated'].includes(r.status)?r.status:selected.status==='pending'?(active?.name===selected.name?r.status:'pending'):selected.status;
  const profiling=variants.map(v=>{
    const profiles=profilesForRun(r,v);
    return profiles.length?`<section class="section">${sectionHead('Profiling evidence',variants.length>1?(v.display_name||v.name):'Studies linked to this experiment; capture settings are recorded on each profile.')}<div class="run-list">${profiles.map(p=>profileCard(p,profileContext(r,v,p))).join('')}</div></section>`:'';
  }).join('');
  const variantNavigation=r.variants.length>1?`<nav class="toolbar" aria-label="Experiment variants"><a class="button small" href="#/run/${encodeURIComponent(r.id)}"${variantName?'':' aria-current="page"'}>All variants</a>${r.variants.map(v=>`<a class="button small" href="${variantLink(r,v)}"${variantName===v.name?' aria-current="page"':''}>${esc(v.name)}</a>`).join('')}</nav><p class="muted">${variantName?`Showing ${esc(variantName)} only.`:`Showing all ${r.variants.length} variants.`}</p>`:'';
  content.innerHTML=heading('Experiment run',variantName||variants.length===1?runDisplayName(r,variants[0]):r.name,r.path,badge(viewStatus))+
    variantNavigation+(r.artifacts.some(a=>a.name==='output-preview.json')?`<p><a class="button small" href="#/output/${r.id}">View PDF → Markdown</a></p>`:'')+
    experimentNote(r,variantName?variants[0]:null)+(r.comparison_context?`<div class="callout">${esc(r.comparison_context.summary)}</div>`:'')+
    (r.verification_context?`<div class="callout">${esc(r.verification_context.summary)}</div>`:'')+
    (r.invalidation?`<div class="callout warning"><strong>Invalidated · ${esc(r.invalidation.model||'OCR completeness failure')}.</strong> ${esc(r.invalidation.reason)} ${esc(r.invalidation.report_note||'These measurements are retained as historical evidence.')}${r.invalidation.followup?`<p>${esc(r.invalidation.followup)}</p>`:''}${r.invalidation.references?.length?`<p>${r.invalidation.references.filter(ref=>ref.url.startsWith('https://')||ref.url.startsWith('#/doc/')).map(ref=>`<a href="${esc(ref.url)}">${esc(ref.label)} ↗</a>`).join(' · ')}</p>`:''}</div>`:'')+
    (r.quality?.status==='incomplete'&&!r.invalidation?`<div class="callout warning"><strong>OCR incomplete — not an accepted baseline.</strong> ${esc(r.quality.summary)} Reference matching passed against truncated outputs.</div>`:'')+
    (r.quality?.status==='complete'?`<div class="callout"><strong>Completeness review finished.</strong> ${esc(r.quality.summary)}${r.quality.known_limitations?.length?`<details><summary>Known OCR errors in the frozen references</summary><ul>${r.quality.known_limitations.map(text=>`<li>${esc(text)}</li>`).join('')}</ul></details>`:''}</div>`:'')+
    `<p class="muted">Per PDF: ${esc(r.experiment.warmup)} warm-up call(s) + ${esc(r.experiment.repeats)} timed call(s).</p>`+
    (r.lifecycle?.phase&&r.lifecycle.phase!=='finished'?`<div class="callout">Phase: <strong>${esc(r.lifecycle.phase)}</strong>${active?` · Currently evaluating ${esc(active.display_name||active.name)}`:''}${r.progress?` · ${r.progress.completed_calls} completed pipeline calls · ${r.progress.completed_pages} pages processed · last completed: ${esc(r.progress.last_document)}`:''}${r.lifecycle.phase==='evaluation'?' · Counts include warm-up calls.':''}</div>`:'')+
    (r.warnings.length?`<div class="callout warning">${r.warnings.map(esc).join('<br>')}</div>`:'')+
    variants.map(perDocumentResults).join('')+
    evaluationSummary(variants)+variants.map(previousComparisonDetails).join('')+
    variants.map(v=>supportingExperiments(r,v)+(v.presentation?.supporting_experiments?.length?'':controlEvidence(r,v))).join('')+
    variants.map(v=>`<section class="section">${sectionHead(v.name+' · median stage timings','Median per timed PDF call · warm-ups excluded')}${v.comparison_note?`<div class="callout ${v.comparison_context?'':'warning'}">${esc(v.comparison_note)}</div>`:''}${v.error?`<div class="error-list">${esc(v.error)}</div>`:''}<div class="card">${charts(v.timing)}</div>${detail('Configuration and environment',{variant:v.config,evaluation:v.configuration,environment:v.environment,dataset_sha256:v.dataset_sha256,cold_start:v.cold_start})}</section>`).join('')+
    profiling+notesFor('run',r.path)+artifacts(r.artifacts)+detail('Experiment definition',r.experiment);
}
function operatorRows() {
  if(!currentProfile)return;
  const term=(document.querySelector('#operator-search')?.value||'').toLowerCase();
  const key=document.querySelector('#operator-sort')?.value||'device_total_us';
  const rows=[...(currentProfile.operators||[])].filter(o=>o.name.toLowerCase().includes(term)).sort((a,b)=>(b[key]||0)-(a[key]||0));
  document.querySelector('#operator-body').innerHTML=rows.slice(0,150).map(o=>`<tr><td class="operator-name mono">${esc(o.name)}</td><td class="number">${num(o.calls,0)}</td><td class="number">${duration(o.cpu_total_us/1000)}</td><td class="number">${duration(o.self_cpu_us==null?null:o.self_cpu_us/1000)}</td><td class="number">${duration(o.device_total_us/1000)}</td><td class="number">${duration(o.self_device_us==null?null:o.self_device_us/1000)}</td></tr>`).join('');
  document.querySelector('#operator-count').textContent=`Showing ${Math.min(rows.length,150)} of ${rows.length} matching operators`;
}
function gpuEvidence(record) {
  const telemetry=record.gpu_telemetry?.ocr,workers=record.ocr_stats?.[0]?.workers||[],shared=record.config?.ocr?.paddle_region_dispatch==='all_gpus';
  if(!telemetry?.devices?.length)return '';
  return `<section class="section">${sectionHead(record.ocr_timing_stage==='render + OCR'?'GPU activity during rendering + OCR':'GPU activity during OCR','One-second samples from the ordinary measured call; the extra instrumented call is separate.')}<div class="table-wrap"><table><thead><tr><th>GPU</th><th>${shared?'Layout pages':'Pages assigned'}</th><th>GPU busy · mean</th><th>Memory busy · mean</th><th>VRAM · peak</th><th>Power · mean</th><th>SM clock · mean</th></tr></thead><tbody>${telemetry.devices.map(d=>{
    const m=d.metrics,w=workers.find(w=>w.worker===d.index);
    return `<tr><td>${esc(d.index)}</td><td>${w?.pages_1_based?.length?esc(w.pages_1_based.join(', ')):shared?'None · recognizes shared regions':'Idle'}</td><td>${num(m.gpu_busy_pct?.mean)}%</td><td>${num(m.memory_busy_pct?.mean)}%</td><td>${num(m.memory_used_mib?.max/1024,2)} GiB</td><td>${num(m.power_w?.mean)} W</td><td>${num(m.sm_clock_mhz?.mean,0)} MHz</td></tr>`;
  }).join('')}</tbody></table></div><p class="tiny muted">Busy percentages measure sampled activity, not percentage of peak compute or memory bandwidth. VRAM capacity and compute utilization are different. Full readings and clock-event flags are in the downloadable telemetry.</p></section>`;
}
function profileFindings(findings) {
  if(!findings)return '';
  return `<section class="section">${sectionHead('Findings and next step')}<div class="callout ${['incomplete','failed'].includes(findings.quality)?'warning':''}"><strong>${esc(findings.summary)}</strong></div><div class="card"><ul>${findings.observations.map(text=>`<li>${esc(text)}</li>`).join('')}</ul><p><strong>Next step:</strong> ${esc(findings.next_step)}</p><p><strong>Optimization experiment:</strong> ${esc(findings.optimization)}</p></div></section>`;
}
function serverEvidence(record) {
  const telemetry=record.vllm_telemetry;
  if(!telemetry)return '';
  const hasWaiting=telemetry.workers.some(w=>Number.isFinite(w.maximum_waiting_requests));
  const speculative=telemetry.workers.filter(w=>w.speculation);
  return `<section class="section">${sectionHead('Region concurrency','vLLM scheduler observations at 100 ms intervals; running requests are not a measure of peak GPU compute.')}<div class="table-wrap"><table><thead><tr><th>GPU</th><th>Maximum running regions</th>${hasWaiting?'<th>Maximum waiting regions</th>':''}<th>Samples</th></tr></thead><tbody>${telemetry.workers.map(w=>`<tr><td>${w.worker}</td><td>${num(w.maximum_running_requests,0)}</td>${hasWaiting?`<td>${num(w.maximum_waiting_requests,0)}</td>`:''}<td>${w.samples}</td></tr>`).join('')}</tbody></table></div></section>`+
    (speculative.length?`<section class="section">${sectionHead('Speculative decoding','Counter changes across ordinary diagnostic calls; warm-up and instrumented calls are excluded. Tokens per step includes the model’s bonus token.')}<div class="table-wrap"><table><thead><tr><th>GPU</th><th>Proposed tokens</th><th>Accepted tokens</th><th>Acceptance</th><th>Tokens per step</th></tr></thead><tbody>${speculative.map(w=>`<tr><td>${w.worker}</td><td>${num(w.speculation.draft_tokens,0)}</td><td>${num(w.speculation.accepted_tokens,0)}</td><td>${num(w.speculation.draft_acceptance_rate*100)}%</td><td>${num(w.speculation.mean_acceptance_length,2)}</td></tr>`).join('')}</tbody></table></div></section>`:'');
}
function profileView(p) {
  currentProfile=p;
  const s=p.summary||{},r=p.record;
  const baseline=profileRun(p);
  content.innerHTML=heading('Diagnostic profile',`${profileName(p)} · ${p.pdf?.name||'Document'}`,p.path,badge(p.status))+
    (baseline&&baseline.path!==p.baseline_run?`<div class="callout">Supporting profile captured ${esc(date(p.created_at))}. Its document and saved configuration are recorded below. Profile timings are separate from this experiment’s evaluator measurements.</div>`:'')+profileFindings(r.findings)+
    `<div class="metrics">${metric('Median call',duration(s.median_total_ms),`${s.calls||0} measured calls`,true)}${metric('Largest stage',s.dominant_stage||'Not available','By mean stage wall time',true)}${metric('OCR generation rate',s.generated_tok_s?`${num(s.generated_tok_s)} tok/s`:'—',s.generated_tok_s_stage==='render + OCR'?'Includes overlapping rendering and OCR':'Includes all OCR work, not isolated decode',true)}</div>`+
    `<div class="callout">Diagnostic measurements are separate from official speedups. Operator totals overlap; use the trace to understand execution.${baseline?` <a href="#/run/${baseline.id}">Open the linked experiment run →</a>`:''}</div>`+
    (p.error?`<div class="callout warning">${esc(p.error)}</div>`:'')+
    `<section class="section">${sectionHead('Where the time goes','Stage measurements exclude warm-up and the extra profiler call.')}<div class="card">${charts(s)}</div></section>`+
    gpuEvidence(r)+serverEvidence(r)+
    (r.hardware_counters?`<div class="callout">Hardware-counter diagnostic: <strong>${esc(r.hardware_counters.status)}</strong> · ${esc(r.hardware_counters.tool)}. ${esc(r.hardware_counters.scope)}.</div>`:'')+
    (r.torch_capture?.maximum_ocr_forwards?`<div class="callout">${r.mode==='paddle'?'Native Paddle tracing records layout work and the first':'The detailed trace samples the first'} ${esc(r.torch_capture.maximum_ocr_forwards)} recognition-model calls per active worker. ${r.mode==='paddle'?'This window can span multiple short regions, each with its own prefill and decoding.':'The sample starts with the first page’s prefill and early decoding.'} The stage measurements above cover the complete PDF.</div>`:'')+
    (r.torch_capture?.maximum_engine_iterations?`<div class="callout">The trace samples the first ${esc(r.torch_capture.maximum_engine_iterations)} engine iterations per active vLLM recognizer. An iteration can process several regions. The stage measurements cover the complete PDF.</div>`:'')+
    (p.operators?.length?`<section class="section">${sectionHead('Operator timings',['paddle','vllm'].includes(r.mode)?'CPU ranges include nested work; GPU kernel rows show device execution. Self time is not computed.':'Inclusive time includes nested work; self time excludes child CPU/device events.')}<div class="toolbar"><input id="operator-search" type="search" placeholder="Filter operators…" aria-label="Filter operators"><select id="operator-sort" aria-label="Sort operators"><option value="device_total_us">Device total time ↓</option><option value="cpu_total_us">CPU total time ↓</option><option value="self_cpu_us">Self CPU time ↓</option><option value="self_device_us">Self device time ↓</option><option value="calls">Call count ↓</option></select></div><div class="table-wrap"><table><thead><tr><th>Operator</th><th>Calls</th><th>CPU total</th><th>CPU self</th><th>Device total</th><th>Device self</th></tr></thead><tbody id="operator-body"></tbody></table></div><p class="tiny muted" id="operator-count"></p></section>`:'')+
    notesFor('profile',p.path)+artifacts(p.artifacts)+detail('Configuration, environment and reproduction command',r);
  if(p.operators?.length)operatorRows();
}
function pipelineDiagram() {
  const cfg=state.defaults;
  const steps=[
    ['Render','PDF → page images','One image per page'],
    ['OCR','Images → formatted text','Read text, tables and formulas'],
    ['Chunk','Text → smaller pieces',`${cfg.chunk.max_tokens}-token target`],
    ['Embed','Text pieces → number lists','Represent the meaning of text']
  ];
  return `<figure class="pipeline-flow" aria-label="Sequential pipeline from PDF pages to embedding vectors"><ol>${steps.map(([name,output,control],i)=>`<li><span class="pipeline-flow-number">0${i+1}</span><strong>${esc(name)}</strong><span class="pipeline-flow-output">${esc(output)}</span><span class="pipeline-flow-control">${esc(control)}</span></li>`).join('')}</ol></figure>`;
}
async function documentView(id,token) {
  const d=await loadJSON(`site-data/docs/${encodePath(id)}.json`); if(token!==generation)return;
  const links=[]; const run=state.runs.find(r=>r.path===d.run),profile=state.profiles.find(p=>p.path===d.profile);
  if(run)links.push(`<a class="button small" href="#/run/${run.id}">Open experiment run ↗</a>`);
  if(profile)links.push(`<a class="button small" href="#/profile/${profile.id}">Open profile ↗</a>`);
  content.innerHTML=heading(d.kind==='experiment'?'Experiment journal':'Notes',d.title,d.description)+
    (d.kind==='experiment'?`<div class="doc-meta">${badge(d.status||'planned')}<span>${esc(d.date)}</span>${links.join('')}</div>`:'')+
    `<div class="doc-layout ${d.id==='pages/pipeline'?'pipeline-doc-layout':''}"><article class="prose">${d.id==='pages/pipeline'?pipelineDiagram():''}${d.html}</article><aside class="toc" aria-label="On this page"><p>ON THIS PAGE</p><div id="toc-items"></div></aside></div>`;
  const headings=[...content.querySelectorAll('.prose h2')];
  headings.forEach((h,i)=>h.id=`section-${i}`);
  document.querySelector('#toc-items').innerHTML=headings.map(h=>`<button type="button" data-scroll="${h.id}">${esc(h.textContent)}</button>`).join('');
  content.querySelectorAll('.prose a[href^="https://"]').forEach(a=>{a.target='_blank';a.rel='noopener noreferrer';});
}
function searchView(term) {
  const q=term.toLowerCase();
  const entries=[...state.documents.map(d=>({...d,url:`#/doc/${d.id}`,category:d.kind==='guide'?'Notes':'Journal'})),...state.runs.map(r=>({title:r.name,description:r.path,url:`#/run/${r.id}`,category:'Run'})),...state.profiles.map(p=>({title:`${p.label} · ${p.pdf?.name}`,description:p.path,url:`#/profile/${p.id}`,category:'Profile'}))].filter(d=>`${d.title} ${d.description}`.toLowerCase().includes(q));
  content.innerHTML=heading('Workspace search',`Results for “${term}”`,`${entries.length} matching pages and recorded artifacts.`)+entries.map(d=>`<a class="resource search-result" href="${d.url}"><div class="eyebrow">${d.category}</div><strong>${esc(d.title)}</strong><p>${esc(d.description)}</p></a>`).join('');
}
async function render() {
  const token=++generation;
  if(!state)return;
  nav();currentProfile=null;currentOutput=null;currentEmbeddings=null;
  const parts=(location.hash||'#/').slice(2).split('/'), route=parts[0];
  if(route==='profiles'){location.replace('#/runs');return;}
  if(route==='doc'&&['pages/ocr-repetition','pages/paddle-repetition'].includes(parts.slice(1).join('/'))){location.replace('#/doc/pages/debugging');return;}
  document.querySelector('#breadcrumb').textContent=`Workspace / ${{'':'Overview',runs:'Experiment runs',run:'Experiment run',output:'PDF → Markdown',embeddings:'Embeddings',profiles:'Profiles',profile:'Profile',datasets:'PDF dataset',dataset:'PDF preview',journal:'Experiment journal',doc:'Notes',search:'Search'}[route]||'Page'}`;
  try {
    if(route==='')overview();else if(route==='runs')runs();else if(route==='journal')journal();
    else if(route==='datasets')datasetsView();
    else if(route==='output')await outputView(parts,token);
    else if(route==='embeddings')await embeddingsView(parts,token);
    else if(route==='dataset')datasetView(parts[1],parts[2],Number(parts[3]||1));
    else if(route==='doc')await documentView(parts.slice(1).join('/'),token);
    else if(route==='run'){const r=await loadJSON(`site-data/runs/${encodeURIComponent(parts[1])}.json`);if(token===generation)runView(r,parts[2]?decodeURIComponent(parts[2]):null);}
    else if(route==='profile'){const p=await loadJSON(`site-data/profiles/${encodeURIComponent(parts[1])}.json`);if(token===generation)profileView(p);}
    else if(route==='search')searchView(decodeURIComponent(parts.slice(1).join('/')));
    else content.innerHTML=empty('Page not found','Choose a page from the navigation.');
    if(token===generation){document.title=`${content.querySelector('h1')?.textContent||'Overview'} · PDF OCR inference`;window.scrollTo(0,0);}
  }catch(e){if(token===generation)content.innerHTML=`<div class="error-state"><h2>Could not load this page</h2><p>${esc(e.message)}</p><button class="button" data-retry>Try again</button></div>`;}
}
async function refresh(notify=false) {
  try{state=await loadJSON('site-data/overview.json');document.querySelector('#updated').textContent=state.exported_at?`Snapshot ${date(state.exported_at)}`:'Saved results';await render();if(notify)toast('Local results and notes refreshed');}
  catch(e){content.innerHTML=`<div class="error-state"><h2>Unable to load results</h2><p>${esc(e.message)}</p><button class="button" data-retry>Try again</button></div>`;}
}

function datasetFile(dataset,document,page=null){const doc=state.datasets.find(d=>d.id===dataset)?.documents.find(d=>d.id===document);return page===null?doc?.pdf_url:doc?.preview_urls?.[page];}
function datasetsView(){
 const datasets=state.datasets||[];
 content.innerHTML=heading('Benchmark inputs','Preview the PDFs','Inspect the exact benchmark inputs. Every PDF has at most five pages.')+
  (datasets.length?datasets.map(d=>`<section class="section">${sectionHead(d.id,`${d.documents.length} benchmark PDFs · ${d.pages} total pages · ${d.path}`)}<div class="pdf-grid">${d.documents.map(doc=>`<a class="pdf-card" href="#/dataset/${encodeURIComponent(d.id)}/${encodeURIComponent(doc.id)}/1"><div class="pdf-thumbnail">${doc.preview_pages.includes(1)?`<img src="${datasetFile(d.id,doc.id,1)}" alt="First page of ${esc(doc.title)}" loading="lazy">`:'<span>Preview not yet generated</span>'}</div><div class="pdf-card-body"><strong>${esc(doc.title)}</strong><p>${esc(doc.category)}</p><div class="run-meta"><span>${doc.pages} pages</span><span>${doc.full_document?'Full document':'Selected pages'}</span></div></div></a>`).join('')}</div></section>`).join(''):`<div class="card">${empty('No selected PDFs yet','Datasets with a manifest under data/ appear here.')}</div>`);
}
function datasetView(datasetId,documentId,pageNumber){
 const dataset=(state.datasets||[]).find(d=>d.id===decodeURIComponent(datasetId));
 if(!dataset)throw new Error('Dataset not found');
 const doc=dataset.documents.find(d=>d.id===decodeURIComponent(documentId));
 if(!doc)throw new Error('Document not found');
 const page=Number.isInteger(pageNumber)?Math.max(1,Math.min(doc.pages,pageNumber)):1;
 const href=n=>`#/dataset/${encodeURIComponent(dataset.id)}/${encodeURIComponent(doc.id)}/${n}`;
 const source=/^https?:\/\//.test(doc.source_url)?`<a class="button small" href="${esc(doc.source_url)}" target="_blank" rel="noopener noreferrer">Publisher source ↗</a>`:'';
 content.innerHTML=heading('Dataset / '+dataset.id,doc.title,doc.category,`<a class="button" href="#/datasets">All PDFs</a>`)+
  `<div class="pdf-view-layout"><aside class="pdf-document-list"><div class="eyebrow">DOCUMENTS</div>${dataset.documents.map(d=>`<a class="pdf-document-link ${d.id===doc.id?'active':''}" href="#/dataset/${encodeURIComponent(dataset.id)}/${encodeURIComponent(d.id)}/1"><strong>${esc(d.name)}</strong><small>${d.pages} pages</small></a>`).join('')}<div class="callout">Preview of the frozen PDF input. Every page is available for inspection.</div></aside><div class="pdf-view-main"><div class="preview-toolbar"><div><a class="button small" href="${href(Math.max(1,page-1))}" aria-label="Previous page">←</a> <select id="preview-page" aria-label="PDF page">${Array.from({length:doc.pages},(_,i)=>`<option value="${href(i+1)}" ${i+1===page?'selected':''}>Page ${i+1} of ${doc.pages}</option>`).join('')}</select> <a class="button small" href="${href(Math.min(doc.pages,page+1))}" aria-label="Next page">→</a></div><span class="tiny muted">Source page ${esc(doc.source_pages[page-1]||page)}</span><button class="button small" data-preview-zoom>Actual size</button></div><div class="preview-canvas">${doc.preview_pages.includes(page)?`<img src="${datasetFile(dataset.id,doc.id,page)}" alt="${esc(doc.title)}, page ${page}, original page ${esc(doc.source_pages[page-1]||page)}">`:empty('Preview not generated','Render this dataset’s previews with scripts/render_pdf_previews.py.')}</div><div class="preview-footer"><a class="button small" href="${datasetFile(dataset.id,doc.id)}" download>Download PDF ↓</a>${source}</div><p class="tiny muted">${esc(doc.notes)}</p><details><summary>Document identity</summary>${pretty({file:`${dataset.path}/pdfs/${doc.name}`,pages:doc.pages,full_document:doc.full_document,...(doc.full_document?{}:{original_page_numbers:doc.source_pages}),sha256:doc.sha256})}</details></div></div>`;
}
window.addEventListener('hashchange',()=>{if(!location.hash.startsWith('#/search'))document.querySelector('#search').value='';render();});
let searchTimer;document.querySelector('#search').addEventListener('input',e=>{clearTimeout(searchTimer);searchTimer=setTimeout(()=>{location.hash=e.target.value.trim()?`#/search/${encodeURIComponent(e.target.value.trim())}`:'#/';},180);});
document.addEventListener('click',e=>{const b=e.target.closest('button');if(!b)return;if(b.hasAttribute('data-run-toggle')){const panel=document.getElementById(b.dataset.runToggle);const open=b.getAttribute('aria-expanded')!=='true';b.setAttribute('aria-expanded',String(open));b.textContent=open?'▾':'▸';b.setAttribute('aria-label',b.getAttribute('aria-label').replace(open?'Expand':'Collapse',open?'Collapse':'Expand'));panel.hidden=!open;if(open)loadRunExpansion(panel);}if(b.hasAttribute('data-run-retry'))loadRunExpansion(document.getElementById(b.dataset.runRetry));if(b.dataset.stage){stage=b.dataset.stage;document.querySelector('#pipeline').innerHTML=stagePanel();}if(b.dataset.scroll)document.getElementById(b.dataset.scroll)?.scrollIntoView({behavior:'smooth'});if(b.hasAttribute('data-preview-zoom')){const canvas=document.querySelector('.preview-canvas');const zoomed=canvas.classList.toggle('actual-size');b.textContent=zoomed?'Fit width':'Actual size';}if(b.hasAttribute('data-retry'))refresh();});
document.addEventListener('input',e=>{if(e.target.id==='operator-search')operatorRows();});
document.addEventListener('change',e=>{if(e.target.id==='operator-sort')operatorRows();if(e.target.id==='preview-page')location.hash=e.target.value;});
document.addEventListener('change',e=>{if(e.target.matches('[data-output-navigation]'))location.hash=e.target.value;});
document.addEventListener('click',e=>{
  if(!e.target.closest('[data-download-markdown]')||!currentOutput)return;
  const url=URL.createObjectURL(new Blob([currentOutput.doc.markdown],{type:'text/markdown;charset=utf-8'}));
  const a=document.createElement('a');a.href=url;a.download=`${currentOutput.doc.id}.md`;a.click();
  setTimeout(()=>URL.revokeObjectURL(url),1000);
});
document.addEventListener('click',event=>{
  if(!currentEmbeddings)return;
  const map=event.target.closest('#embedding-map');
  if(map){
    const cursor=new DOMPoint(event.clientX,event.clientY).matrixTransform(map.getScreenCTM().inverse());
    const nearest=embeddingVisible().map(index=>({index,distance:Math.hypot(currentEmbeddings.positions[index][0]-cursor.x,currentEmbeddings.positions[index][1]-cursor.y)})).sort((a,b)=>a.distance-b.distance)[0];
    if(nearest&&nearest.distance<=14)selectEmbedding(nearest.index);
    return;
  }
  const target=event.target.closest('[data-embedding-select],[data-embedding-document],[data-embedding-zoom]');
  if(!target)return;
  if(target.hasAttribute('data-embedding-select'))selectEmbedding(Number(target.dataset.embeddingSelect));
  if(target.hasAttribute('data-embedding-document')){
    currentEmbeddings.document=target.dataset.embeddingDocument;updateEmbeddings();
  }
  if(target.hasAttribute('data-embedding-zoom')){
    const action=target.dataset.embeddingZoom;
    currentEmbeddings.zoom=action==='reset'?1:Math.max(1,Math.min(4,currentEmbeddings.zoom*(action==='in'?1.4:1/1.4)));updateEmbeddings();
  }
});
document.addEventListener('keydown',event=>{
  const point=event.target.closest('.embedding-node');
  if(point&&['Enter',' '].includes(event.key)){
    event.preventDefault();const index=Number(point.dataset.embeddingSelect);selectEmbedding(index);
    document.querySelector(`.embedding-node[data-embedding-select="${index}"]`)?.focus({preventScroll:true});
  }
});
document.addEventListener('input',event=>{
  if(event.target.id==='embedding-text-filter'&&currentEmbeddings){currentEmbeddings.search=event.target.value.trim().toLowerCase();updateEmbeddings();}
});
document.addEventListener('change',event=>{
  if(!currentEmbeddings)return;
  if(event.target.id==='embedding-chunk')selectEmbedding(Number(event.target.value));
  if(event.target.id==='embedding-other-pdfs'){currentEmbeddings.otherDocuments=event.target.checked;updateEmbeddings();}
});
refresh();
