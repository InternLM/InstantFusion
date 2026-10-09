'use strict';

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');

// Original results only: checkpoint filenames refer to OPD training updates.
const examples = {
  '02': 'A fox with a green backpack in a snowy pine forest.',
  '08': 'A scientist examining a glowing blue crystal.',
  '11': 'A small astronaut among giant pink flowers.'
};
let currentCase = '02';
let timer = null;
const checkpointSlider = $('#checkpoint-slider');
function setCheckpoint() {
  const step = (Number(checkpointSlider.value) + 1) * 300;
  $('#checkpoint-image').src = `assets/p${currentCase}-step_${String(step).padStart(6,'0')}.webp`;
  $('#checkpoint-image').alt = `Cross-model OPD at training step ${step}: ${examples[currentCase]}`;
  $('#checkpoint-label').textContent = `STEP ${step.toLocaleString('en-US')}`;
  checkpointSlider.setAttribute('aria-valuetext', `Training step ${step}`);
  checkpointSlider.style.setProperty('--fill', `${Number(checkpointSlider.value)/7*100}%`);
}
function stopPlayback() {
  clearInterval(timer); timer = null;
  $('#play-timeline').textContent = '▶';
  $('#play-timeline').setAttribute('aria-label', 'Play training checkpoints');
}
checkpointSlider.addEventListener('input', () => {stopPlayback(); setCheckpoint();});
$$('[data-case]').forEach(button => button.addEventListener('click', () => {
  currentCase = button.dataset.case;
  $$('[data-case]').forEach(b => {b.classList.toggle('selected', b === button); b.setAttribute('aria-pressed', String(b === button));});
  $('#baseline-image').src = `assets/p${currentCase}-baseline.webp`;
  $('#baseline-image').alt = `SD3 baseline: ${examples[currentCase]}`;
  $('#case-caption').textContent = examples[currentCase];
  setCheckpoint();
}));
$('#play-timeline').addEventListener('click', () => {
  if (timer) {stopPlayback(); return;}
  if (Number(checkpointSlider.value) === 7) {checkpointSlider.value = 0; setCheckpoint();}
  $('#play-timeline').textContent = 'Ⅱ';
  $('#play-timeline').setAttribute('aria-label', 'Pause training checkpoints');
  timer = setInterval(() => {
    if (Number(checkpointSlider.value) >= 7) {stopPlayback(); return;}
    checkpointSlider.value = Number(checkpointSlider.value) + 1;
    setCheckpoint();
  }, 1000);
});
document.addEventListener('visibilitychange', () => {if(document.hidden) stopPlayback();});

const metricData = {
  acceleration: {
    caption: 'Generation quality and inference efficiency',
    columns: ['Model / combination', 'Time (s) ↓', 'GenEval-v2 ↑', 'CLIP ↑', 'OneIG avg. ↑'],
    rows: [
      ['Qwen-Image','18.840','41.73','25.46','0.4849'],
      ['Qwen-Few-Step','0.893','32.94','27.48','0.5148'],
      ['FLUX.2-Klein','0.516','25.43','27.71','0.3597'],
      ['Z-Image-Turbo','3.521','39.21','25.02','0.5669'],
      ['T-Stitch','20.760','7.762','21.24','0.2659'],
      ['Z-Image-Turbo + Qwen-Image','4.373','33.34','27.90','0.5386',true],
      ['FLUX.2-Klein + Qwen-Image','2.545','27.19','29.18','0.5467',true],
      ['Qwen-Few-Step + Qwen-Image','10.640','32.70','33.15','0.5204',true],
      ['Qwen-Few-Step + Qwen-Image + DiCache','5.700','32.46','33.18','0.5220',true]
    ],
    highlights: [['18.84 → 2.545 <small>s</small>','Qwen-Image vs. FLUX.2-Klein + Qwen-Image'],['10.64 → 5.70 <small>s</small>','Qwen-Few-Step + Qwen-Image, with DiCache']],
    note: 'Time is measured in seconds. Lower latency is better; higher quality scores are better.'
  },
  quality: {
    caption: 'Cross-model transfer with Qwen-Image',
    columns: ['Model / combination', 'GenEval-v2 ↑', 'CVTG-2K ↑', 'OneIG avg. ↑'],
    rows: [
      ['Qwen-Image','41.73','0.6699','0.4849'],
      ['SD3','22.14','0.3682','0.4157'],
      ['FLUX.1','22.86','0.5180','0.4416'],
      ['Qwen-Image + SD3','35.06 <small>+12.92</small>','0.6488 <small>+0.2806</small>','0.4776 <small>+0.0619</small>',true],
      ['Qwen-Image + FLUX.1','36.54 <small>+13.68</small>','0.6983 <small>+0.1803</small>','0.5007 <small>+0.0591</small>',true]
    ],
    highlights: [['+12.92','GenEval-v2 improvement over the SD3 baseline'],['+13.68','GenEval-v2 improvement over the FLUX.1 baseline']],
    note: 'Improvements are measured against the corresponding SD3 or FLUX.1 baseline. Cross-model combinations improve these baselines; they do not exceed Qwen-Image on every metric.'
  }
};
function renderMetrics(key, tableId, highlightsId, noteId) {
  const data = metricData[key];
  const groups = [
    ['Generation baselines', data.rows.filter(row => row.at(-1) !== true)],
    ['InstantFusion', data.rows.filter(row => row.at(-1) === true)]
  ];
  document.getElementById(tableId).innerHTML = '<caption>'+data.caption+'</caption><thead><tr>'+data.columns.map(c=>'<th scope="col">'+c+'</th>').join('')+'</tr></thead>'+groups.map(([label,rows])=>'<tbody aria-label="'+label+'"><tr class="metric-group"><th colspan="5" scope="rowgroup">'+label+'</th></tr>'+rows.map(row=>'<tr class="metric-data'+(row.at(-1)===true?' ours':'')+'"><th scope="row">'+row[0].replace(' + Qwen-Image',' <span class="model-partner">+ Qwen-Image</span>').replace(' + DiCache',' <span class="cache-tag">+ DiCache</span>')+'</th>'+row.slice(1,5).map((v,i)=>'<td'+(i===0?' class="latency"':'')+'>'+v+'</td>').join('')+'</tr>').join('')+'</tbody>').join('');
  document.getElementById(highlightsId).innerHTML=data.highlights.map(([value,label])=>'<div><strong>'+value+'</strong><p>'+label+'</p></div>').join('');
  document.getElementById(noteId).textContent=data.note;
}
renderMetrics('acceleration','metrics-table','metric-highlights','metric-note');
function renderTransferComparison() {
  const data = metricData.quality;
  const labels = ['GenEval-v2', 'CVTG-2K', 'OneIG avg.'];
  document.getElementById('quality-pairs').innerHTML = [1,2].map((baseIndex,index) => {
    const baseline=data.rows[baseIndex], combined=data.rows[baseIndex+2];
    return '<div class="quality-pair"><span class="mono">0'+(index+1)+' / CROSS-MODEL TRANSFER</span><table><caption>'+baseline[0]+' <span>+ Qwen-Image</span></caption><thead><tr><th scope="col">Metric ↑</th><th scope="col">Baseline</th><th scope="col">Combined</th><th scope="col">Gain</th></tr></thead><tbody>'+labels.map((label,i)=>{
      const match=combined[i+1].match(/^([^<]+) <small>([^<]+)<\/small>$/);
      return '<tr><th scope="row">'+label+'</th><td>'+baseline[i+1]+'</td><td class="combined-score">'+match[1]+'</td><td class="gain-score">'+match[2]+'</td></tr>';
    }).join('')+'</tbody></table></div>';
  }).join('');
  document.getElementById('quality-note').textContent='Gains are measured against each corresponding baseline.';
}
renderTransferComparison();

const dialog = $('#figure-dialog');
$$('.figure-open').forEach(button => button.addEventListener('click', () => {
  $('#dialog-image').src = button.dataset.image;
  $('#dialog-image').alt = button.dataset.caption;
  $('#dialog-caption').textContent = button.dataset.caption;
  dialog.showModal(); document.body.classList.add('dialog-open');
}));
$('#close-dialog').addEventListener('click',()=>dialog.close());
dialog.addEventListener('click',event=>{if(event.target === dialog && (event.clientX < dialog.getBoundingClientRect().left || event.clientX > dialog.getBoundingClientRect().right || event.clientY < dialog.getBoundingClientRect().top || event.clientY > dialog.getBoundingClientRect().bottom)) dialog.close();});
dialog.addEventListener('close',()=>document.body.classList.remove('dialog-open'));

// The readable DOM title dissolves into a sampled typographic particle field.
// Scroll position is the clock: no permanent animation loop or per-frame layout work.
const hero = $('.hero');
const composition = $('.hero-composition');
const title = $('#hero-title');
const canvas = $('#title-particles');
const context = canvas.getContext('2d');
const rasterArt = $('.raster-art');
let particles = [];
let size = {width:0,height:0,top:0,travel:1};
let framePending = false;
let seed = 12345;
function random(){seed=(seed*16807)%2147483647;return (seed-1)/2147483646;}
function sampleTitle() {
  if (!context) return;
  const box = composition.getBoundingClientRect();
  size = {width:box.width,height:box.height,top:hero.offsetTop,travel:Math.max(1,hero.offsetHeight-window.innerHeight)};
  const dpr = Math.min(window.devicePixelRatio || 1,2);
  canvas.width = Math.round(box.width*dpr); canvas.height=Math.round(box.height*dpr);
  context.setTransform(dpr,0,0,dpr,0,0);
  particles=[];seed=12345;
  const mask=document.createElement('canvas');mask.width=Math.ceil(box.width);mask.height=Math.ceil(box.height);
  const ctx=mask.getContext('2d',{willReadFrequently:true});
  const titleStyle=getComputedStyle(title);
  const step=window.innerWidth<760?5:6;
  [...title.children].forEach((line,index)=>{
    const bounds=line.getBoundingClientRect();
    ctx.clearRect(0,0,mask.width,mask.height);
    ctx.font=`${titleStyle.fontWeight} ${titleStyle.fontSize} ${titleStyle.fontFamily}`;
    ctx.textBaseline='top';ctx.fillStyle='#fff';
    if('letterSpacing' in ctx)ctx.letterSpacing=titleStyle.letterSpacing;
    // Canvas and DOM fonts share the same serif; optical baseline adjustment.
    ctx.fillText(index===0?'Instant':'Fusion',bounds.left-box.left,bounds.top-box.top-Number.parseFloat(titleStyle.fontSize)*.07);
    const data=ctx.getImageData(0,0,mask.width,mask.height).data;
    for(let y=0;y<mask.height;y+=step)for(let x=0;x<mask.width;x+=step){
      if(data[(y*mask.width+x)*4+3]>85){particles.push({x,y,dx:(random()-.42)*size.width*.8,dy:(random()-.65)*size.height*1.3,delay:random()*.24,r:random()*1.3+.6,char:['·',':','+','/','○'][Math.floor(random()*5)],ascii:random()>.77,color:index===0?'#284bcd':'#172323'});}
    }
  });
  drawHero();
}
function drawHero(){
  framePending=false;
  if(!context)return;
  context.clearRect(0,0,size.width,size.height);
  if(reducedMotion.matches){
    title.style.opacity='1';rasterArt.style.transform='';rasterArt.style.opacity='1';
    ['.hero-bottom','.hero-topline','.hero-foot'].forEach(selector=>$(selector).style.opacity='1');
    return;
  }
  const p=Math.max(0,Math.min(1,(window.scrollY-size.top)/size.travel));
  title.style.opacity=String(1-Math.min(1,p/.23));
  rasterArt.style.transform=`translateY(${-p*55}px) scale(${1+p*.08})`;
  rasterArt.style.opacity=String(Math.max(0,1-p*1.2));
  $('.hero-bottom').style.opacity=String(Math.max(0,1-p*2));
  $('.hero-topline').style.opacity=String(Math.max(0,1-p*1.8));
  $('.hero-foot').style.opacity=String(Math.max(0,1-p*2));
  if(p<=.015||p>=1)return;
  const enter=Math.min(1,p/.16);
  context.font='11px "Courier New",monospace';
  for(const particle of particles){
    const t=Math.max(0,(p-particle.delay*.4)/.92);
    const ease=t*t*(3-2*t);
    context.globalAlpha=enter*Math.max(0,1-Math.pow(t,1.35));
    context.fillStyle=particle.color;
    const x=particle.x+particle.dx*ease,y=particle.y+particle.dy*ease;
    if(particle.ascii && p>.19)context.fillText(particle.char,x,y);
    else{context.beginPath();context.arc(x,y,particle.r,0,Math.PI*2);context.fill();}
  }
  context.globalAlpha=1;
}
function scheduleDraw(){if(!framePending){framePending=true;requestAnimationFrame(drawHero);}}
window.addEventListener('scroll',scheduleDraw,{passive:true});
let resizeTimer;
window.addEventListener('resize',()=>{clearTimeout(resizeTimer);resizeTimer=setTimeout(sampleTitle,120);});
reducedMotion.addEventListener('change',sampleTitle);
document.fonts.ready.then(sampleTitle);
setCheckpoint();
