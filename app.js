import {FilesetResolver, FaceDetector} from './vision_bundle.mjs';
const VERSION='v4';
const $=id=>document.getElementById(id);
const video=$('video'),overlay=$('overlay'),octx=overlay.getContext('2d');
const params=new URLSearchParams(location.search);
const S={facing:'environment',stream:null,dets:{},range:'auto',tracks:[],nextId:1,rec:null,videoMode:false,
  delegate:'–',fps:0,detMs:0,detFps:0,inputMode:params.get('input')||'canvas',lastError:'',status:'',
  stale:0,staleSwitches:0,inputChanging:null,lastCT:0,ctAdvancing:null,fs:null};
window.__stats=S;
const INPUT_MODES=['canvas','bitmap','video'];

function toast(t,ms=2200){const e=$('toast');e.textContent=t;e.classList.add('show');clearTimeout(toast.t);toast.t=setTimeout(()=>e.classList.remove('show'),ms);}
function err(where,e){S.lastError=`${new Date().toLocaleTimeString()} ${where}: ${e&&e.message||e}`;console.warn(where,e);}

// ---------- détecteur ----------
async function makeDetector(model,delegate){
  return FaceDetector.createFromOptions(S.fs,{baseOptions:{modelAssetPath:'models/'+model,delegate},
    runningMode:'VIDEO',minDetectionConfidence:0.3,minSuppressionThreshold:0.3});
}
async function buildDetectors(delegate){
  for(const k in S.dets){try{S.dets[k].close();}catch(e){}}S.dets={};
  S.dets.short=await makeDetector('blaze_face_short_range.tflite',delegate);
  try{S.dets.full=await makeDetector('blaze_face_full_range.tflite',delegate);}catch(e){err('full-range',e);}
  S.delegate=delegate;
}
async function initDetectors(){
  S.status='Chargement du détecteur…';
  S.fs=await FilesetResolver.forVisionTasks(new URL('wasm',location.href).href.replace(/\/$/,''));
  for(const d of (params.get('cpu')?['CPU']:['GPU','CPU'])){
    try{await buildDetectors(d);S.status='';return;}catch(e){err('délégué '+d,e);}
  }
  S.status='Détection indisponible';
}
async function toCPU(reason){
  if(S.delegate==='CPU'||S.switching)return false;S.switching=true;S.status='Bascule CPU ('+reason+')';
  try{await buildDetectors('CPU');S.status='';}catch(e){err('CPU',e);S.status='Détection indisponible';}
  S.switching=false;return true;
}

// ---------- caméra ----------
async function startCamera(){
  if(S.stream)S.stream.getTracks().forEach(t=>t.stop());
  S.stream=await navigator.mediaDevices.getUserMedia({audio:false,video:{facingMode:{ideal:S.facing},width:{ideal:1280},height:{ideal:720},frameRate:{ideal:30}}});
  video.muted=true;video.playsInline=true;video.setAttribute('playsinline','');video.setAttribute('muted','');video.setAttribute('autoplay','');
  video.classList.toggle('mirror',S.facing==='user');
  video.srcObject=S.stream;
  await new Promise(r=>{if(video.readyState>=1)return r();video.addEventListener('loadedmetadata',r,{once:true});setTimeout(r,3000);});
  try{await video.play();}catch(e){err('play()',e);needTap();}
  S.tracks=[];
}
function needTap(){toast('Touchez l’écran pour démarrer la vidéo',4000);
  const f=()=>video.play().catch(e=>err('play() tap',e));
  document.addEventListener('touchend',f,{once:true});document.addEventListener('click',f,{once:true});}

// ---------- suivi des boîtes (coordonnées vidéo) ----------
function iou(a,b){const x1=Math.max(a.x,b.x),y1=Math.max(a.y,b.y),x2=Math.min(a.x+a.w,b.x+b.w),y2=Math.min(a.y+a.h,b.y+b.h);
  const i=Math.max(0,x2-x1)*Math.max(0,y2-y1);return i/(a.w*a.h+b.w*b.h-i||1);}
function headBox(x,y,bw,bh){const cx=x+bw/2,cy=y+bh/2,s=Math.max(bw,bh),w=s*1.55,h=s*1.85;return {x:cx-w/2,y:cy-h*0.58,w,h};}
function nms(b){b.sort((p,q)=>q.score-p.score);const o=[];for(const x of b)if(!o.some(y=>iou(x,y)>0.3))o.push(x);return o;}
function updateTracks(boxes){
  const used=new Set();
  for(const t of S.tracks){let best=-1,bi=0;
    boxes.forEach((b,i)=>{if(used.has(i))return;const v=iou(t,b)+(Math.hypot(t.x+t.w/2-b.x-b.w/2,t.y+t.h/2-b.y-b.h/2)<t.w*0.6?0.2:0);if(v>bi){bi=v;best=i;}});
    if(best>=0&&bi>0.1){const b=boxes[best],a=0.6;used.add(best);const nx=t.x+(b.x-t.x)*a,ny=t.y+(b.y-t.y)*a;
      t.vx=nx-t.x;t.vy=ny-t.y;t.x=nx;t.y=ny;t.w+=(b.w-t.w)*a;t.h+=(b.h-t.h)*a;t.miss=0;}
    else{t.miss++;t.x+=t.vx*0.5;t.y+=t.vy*0.5;}}
  boxes.forEach((b,i)=>{if(!used.has(i))S.tracks.push({...b,vx:0,vy:0,miss:0,id:S.nextId++});});
  S.tracks=S.tracks.filter(t=>t.miss<=8);
}
const boxesOf=()=>S.tracks.map(t=>{const p=t.miss?0.08*t.w:0;return [t.x-p,t.y-p,t.w+2*p,t.h+2*p];});

// ---------- détection (boucle séparée, isolée) ----------
const small=document.createElement('canvas'),sctx=small.getContext('2d',{willReadFrequently:true});
const probe=document.createElement('canvas');probe.width=16;probe.height=9;const pctx=probe.getContext('2d',{willReadFrequently:true});
let lastTs=0,frameN=0,lastHash=null,errCount=0,detN=0;
function hashOf(src){pctx.drawImage(src,0,0,16,9);const d=pctx.getImageData(0,0,16,9).data;let h=0;for(let i=0;i<d.length;i+=4)h=(h*31+d[i]+d[i+1]*7+d[i+2]*13)|0;return h;}
async function detectOnce(){
  if(params.get('faildetect'))throw new Error('échec simulé');
  const vw=video.videoWidth,vh=video.videoHeight;if(!vw||video.readyState<2)return;
  const which=S.range==='proche'?'short':S.range==='loin'?(S.dets.full?'full':'short'):(S.dets.full&&frameN%3===2?'full':'short');
  const det=S.dets[which]||S.dets.short;if(!det)return;frameN++;
  let input,sc=1,bitmap=null;
  if(S.inputMode==='video'){input=video;}
  else if(S.inputMode==='bitmap'){sc=Math.min(1,640/Math.max(vw,vh));
    bitmap=await createImageBitmap(video,{resizeWidth:Math.round(vw*sc),resizeHeight:Math.round(vh*sc)});input=bitmap;}
  else{sc=Math.min(1,640/Math.max(vw,vh));const w=Math.round(vw*sc),h=Math.round(vh*sc);
    if(small.width!==w||small.height!==h){small.width=w;small.height=h;}sctx.drawImage(video,0,0,w,h);input=small;}
  // contrôle « image figée » : l'entrée de détection change-t-elle alors que currentTime avance ?
  try{const h=hashOf(input);const ct=video.currentTime;const adv=ct>S.lastCT;S.ctAdvancing=adv;S.lastCT=ct;
    S.inputChanging=(h!==lastHash);
    if(h===lastHash&&adv){if(++S.stale>=15){S.stale=0;S.staleSwitches++;
      err('entrée figée',S.inputMode+' renvoie toujours la même image');
      S.inputMode=INPUT_MODES[(INPUT_MODES.indexOf(S.inputMode)+1)%3];
      if(S.staleSwitches===3)toCPU('image figée');}}
    else S.stale=0;lastHash=h;}catch(e){err('sonde',e);}
  let ts=performance.now();if(ts<=lastTs)ts=lastTs+1;lastTs=ts;
  const t0=performance.now();
  try{const r=det.detectForVideo(input,ts);
    S.detMs=S.detMs*0.9+(performance.now()-t0)*0.1;detN++;
    updateTracks(nms(r.detections.map(d=>{const b=d.boundingBox;return {...headBox(b.originX/sc,b.originY/sc,b.width/sc,b.height/sc),score:d.categories[0]?.score||0};})));
  }finally{if(bitmap)bitmap.close();}
}
async function detectLoop(){
  if(!S.switching&&S.dets.short&&!document.hidden){
    try{await detectOnce();errCount=0;if(S.status.startsWith('Détection en erreur'))S.status='';}
    catch(e){errCount++;err('détection',e);S.status='Détection en erreur — vidéo maintenue';
      for(const t of S.tracks)t.miss++;S.tracks=S.tracks.filter(t=>t.miss<=8);
      if(errCount>=3&&S.delegate==='GPU')toCPU('erreurs');}
  }
  setTimeout(detectLoop,Math.min(150,Math.max(15,S.detMs*0.5))); // throttle : laisse respirer le thread principal
}

// ---------- overlay (uniquement les carrés noirs) ----------
function coverMap(){ // vidéo -> écran, object-fit: cover
  const cw=overlay.clientWidth,ch=overlay.clientHeight,vw=video.videoWidth,vh=video.videoHeight;
  const s=Math.max(cw/vw,ch/vh);return {s,ox:(cw-vw*s)/2,oy:(ch-vh*s)/2,cw,ch};
}
let fpsN=0,fpsT=performance.now(),lastCTr=-1;
function drawOverlay(){
  requestAnimationFrame(drawOverlay);
  const dpr=window.devicePixelRatio||1,W=Math.round(overlay.clientWidth*dpr),H=Math.round(overlay.clientHeight*dpr);
  if(overlay.width!==W||overlay.height!==H){overlay.width=W;overlay.height=H;}
  octx.setTransform(dpr,0,0,dpr,0,0);octx.clearRect(0,0,W,H);
  if(video.videoWidth){const m=coverMap(),mir=S.facing==='user';octx.fillStyle='#000';
    for(const [x,y,w,h] of boxesOf()){let X=m.ox+x*m.s;const Y=m.oy+y*m.s,Wd=w*m.s,Hd=h*m.s;if(mir)X=m.cw-X-Wd;octx.fillRect(X,Y,Wd,Hd);}}
  if(video.currentTime!==lastCTr){lastCTr=video.currentTime;fpsN++;}
  const now=performance.now();
  if(now-fpsT>1000){S.fps=fpsN*1000/(now-fpsT);S.detFps=detN*1000/(now-fpsT);fpsN=0;detN=0;fpsT=now;
    $('stats').textContent=`${VERSION} · ${S.fps.toFixed(0)} i/s · détection ${S.detMs.toFixed(0)} ms · ${S.delegate} · ${S.tracks.length} visage(s)`+(S.status?'\n⚠ '+S.status:'');}
}

// ---------- diagnostic ----------
let debugOn=!!params.get('debug'),taps=0,tapT=0;
$('stats').addEventListener('click',()=>{const n=Date.now();taps=n-tapT<600?taps+1:1;tapT=n;if(taps>=5){taps=0;debugOn=!debugOn;$('debug').hidden=!debugOn;}});
$('debug').hidden=!debugOn;
setInterval(()=>{if(!debugOn)return;const tr=S.stream?.getVideoTracks()[0];let set={};try{set=tr?.getSettings()||{};}catch(e){}
  $('debug').textContent=[`Caméra Masque ${VERSION}`,navigator.userAgent,
    `standalone: ${navigator.standalone||matchMedia('(display-mode: standalone)').matches}`,
    `piste: ${tr?`${tr.readyState} muted=${tr.muted} enabled=${tr.enabled} ${set.width}x${set.height}@${set.frameRate} ${set.facingMode||''}`:'aucune'}`,
    `video: readyState=${video.readyState} paused=${video.paused} t=${video.currentTime.toFixed(2)} avance=${S.ctAdvancing} ${video.videoWidth}x${video.videoHeight}`,
    `affichage: ${S.fps.toFixed(1)} i/s (nouvelles images vidéo)`,
    `détection: ${S.delegate} entrée=${S.inputMode} ${S.detMs.toFixed(0)} ms ${S.detFps.toFixed(1)}/s`,
    `pixels d'entrée changent: ${S.inputChanging} figée×${S.stale} bascules=${S.staleSwitches}`,
    `visages: ${S.tracks.length}  statut: ${S.status||'ok'}`,`dernière erreur: ${S.lastError||'aucune'}`,
    `SW: ${navigator.serviceWorker?.controller?'actif':'non'}`].join('\n');},500);

// ---------- capture (composition uniquement au moment voulu) ----------
const comp=document.createElement('canvas'),cctx=comp.getContext('2d');
function composite(){const vw=video.videoWidth,vh=video.videoHeight;if(comp.width!==vw||comp.height!==vh){comp.width=vw;comp.height=vh;}
  cctx.save();if(S.facing==='user'){cctx.translate(vw,0);cctx.scale(-1,1);}
  cctx.drawImage(video,0,0,vw,vh);cctx.fillStyle='#000';for(const [x,y,w,h] of boxesOf())cctx.fillRect(x,y,w,h);cctx.restore();}
async function saveFile(blob,name){
  const file=new File([blob],name,{type:blob.type});
  if(navigator.canShare&&navigator.canShare({files:[file]})){try{await navigator.share({files:[file],title:'Caméra Masque'});return;}catch(e){if(e.name==='AbortError')return;err('partage',e);}}
  const a=$('dl');a.href=URL.createObjectURL(blob);a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),30000);toast('Fichier téléchargé');
}
const stamp=()=>new Date().toISOString().replace(/[:.]/g,'-').slice(0,19);
async function takePhoto(){
  if(!video.videoWidth)return;composite();
  document.body.classList.remove('flash');void document.body.offsetWidth;document.body.classList.add('flash');
  const blob=await new Promise(r=>comp.toBlob(r,'image/jpeg',0.92));await saveFile(blob,`masque-${stamp()}.jpg`);
}
function pickMime(){for(const m of['video/mp4;codecs=avc1','video/mp4','video/webm;codecs=vp9','video/webm'])if(window.MediaRecorder&&MediaRecorder.isTypeSupported(m))return m;return '';}
function toggleRec(){
  if(S.rec){S.rec.stop();return;}
  if(!window.MediaRecorder||!comp.captureStream){toast('Enregistrement vidéo non pris en charge sur cet appareil');return;}
  composite();const mime=pickMime(),chunks=[];
  const rec=new MediaRecorder(comp.captureStream(30),mime?{mimeType:mime,videoBitsPerSecond:6e6}:undefined);
  let on=true;const tick=()=>{if(!on)return;composite();requestAnimationFrame(tick);};tick();
  rec.ondataavailable=e=>e.data.size&&chunks.push(e.data);
  rec.onstop=async()=>{on=false;S.rec=null;document.body.classList.remove('rec');const type=rec.mimeType||mime||'video/mp4';
    await saveFile(new Blob(chunks,{type}),`masque-${stamp()}.${type.includes('mp4')?'mp4':'webm'}`);};
  rec.start(1000);S.rec=rec;document.body.classList.add('rec');toast('Enregistrement…',1200);
}

// ---------- UI ----------
$('version').textContent=VERSION;
$('shutter').onclick=()=>S.videoMode?toggleRec():takePhoto();
$('modeBtn').onclick=()=>{if(S.rec)return;S.videoMode=!S.videoMode;document.body.classList.toggle('video',S.videoMode);$('modeBtn').textContent=S.videoMode?'Photo':'Vidéo';};
$('flipBtn').onclick=async()=>{if(S.rec)return;S.facing=S.facing==='user'?'environment':'user';try{await startCamera();}catch(e){err('caméra',e);toast('Caméra indisponible : '+e.message);}};
const RANGES=['auto','proche','loin'];
$('rangeBtn').onclick=()=>{S.range=RANGES[(RANGES.indexOf(S.range)+1)%3];$('rangeBtn').textContent='Portée : '+S.range;S.tracks=[];};
let loopsStarted=false;
async function start(){
  const msg=$('msg');msg.textContent='';$('startBtn').disabled=true;$('startBtn').textContent='Chargement…';
  try{
    if(!navigator.mediaDevices?.getUserMedia)throw new Error('Caméra non accessible (HTTPS requis, ou ouvrez dans Safari).');
    await startCamera();$('start').hidden=true;
    if(!loopsStarted){loopsStarted=true;requestAnimationFrame(drawOverlay);detectLoop();}
    if(!S.dets.short&&!S.initing){S.initing=true;initDetectors().catch(e=>{err('init',e);S.status='Détection indisponible';}).finally(()=>S.initing=false);}
  }catch(e){err('démarrage',e);
    msg.textContent=e.name==='NotAllowedError'?'Accès à la caméra refusé. Réglages › Safari › Caméra › Autoriser (ou Réglages › Caméra Masque), puis réessayez.':'Erreur : '+(e.message||e);
    $('startBtn').disabled=false;$('startBtn').textContent='Réessayer';}
}
$('startBtn').onclick=start;
function resume(){if(document.hidden||!S.stream)return;
  if(S.stream.getVideoTracks().every(t=>t.readyState==='ended'))startCamera().catch(e=>{err('reprise',e);$('start').hidden=false;$('startBtn').disabled=false;$('startBtn').textContent='Réactiver la caméra';});
  else if(video.paused)video.play().catch(()=>needTap());}
document.addEventListener('visibilitychange',resume);window.addEventListener('pageshow',resume);
if('serviceWorker' in navigator){navigator.serviceWorker.register('sw.js',{updateViaCache:'none'}).then(r=>r.update()).catch(e=>err('SW',e));
  let reloaded=false;navigator.serviceWorker.addEventListener('controllerchange',()=>{if(!reloaded&&!S.stream){reloaded=true;location.reload();}});}
if(params.get('autostart'))start();
