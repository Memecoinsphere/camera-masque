import {FilesetResolver, FaceDetector} from './vision_bundle.mjs';
const $=id=>document.getElementById(id);
const video=$('video'),canvas=$('canvas'),ctx=canvas.getContext('2d',{alpha:false});
const S={facing:'environment',stream:null,dets:{},range:'auto',frame:0,tracks:[],nextId:1,
  rec:null,videoMode:false,delegate:'?',fps:0,detMs:0,detFps:0};
const RANGES=['auto','proche','loin'];
const params=new URLSearchParams(location.search);
window.__stats=S; // pour les tests

function toast(t,ms=2200){const e=$('toast');e.textContent=t;e.classList.add('show');clearTimeout(toast.t);toast.t=setTimeout(()=>e.classList.remove('show'),ms);}

async function makeDetector(model,delegate,fs){
  return FaceDetector.createFromOptions(fs,{baseOptions:{modelAssetPath:'models/'+model,delegate},
    runningMode:'VIDEO',minDetectionConfidence:0.3,minSuppressionThreshold:0.3});
}
async function initDetectors(){
  const fs=await FilesetResolver.forVisionTasks(new URL('wasm',location.href).href.replace(/\/$/,''));
  const want=params.get('cpu')?['CPU']:['GPU','CPU'];
  for(const d of want){
    try{
      S.dets.short=await makeDetector('blaze_face_short_range.tflite',d,fs);
      try{S.dets.full=await makeDetector('blaze_face_full_range.tflite',d,fs);}catch(e){console.warn('full-range indisponible',e);}
      S.delegate=d;return;
    }catch(e){console.warn('délégué '+d+' échoué',e);}
  }
  throw new Error('Impossible de charger le détecteur');
}

async function startCamera(){
  if(S.stream)S.stream.getTracks().forEach(t=>t.stop());
  const c={audio:false,video:{facingMode:{ideal:S.facing},width:{ideal:1280},height:{ideal:720},frameRate:{ideal:30}}};
  S.stream=await navigator.mediaDevices.getUserMedia(c);
  video.srcObject=S.stream;
  await video.play();
  S.tracks=[];
}

// ---- suivi des boîtes ----
function iou(a,b){const x1=Math.max(a.x,b.x),y1=Math.max(a.y,b.y),x2=Math.min(a.x+a.w,b.x+b.w),y2=Math.min(a.y+a.h,b.y+b.h);
  const i=Math.max(0,x2-x1)*Math.max(0,y2-y1);return i/(a.w*a.h+b.w*b.h-i||1);}
function headBox(bb){ // agrandit la boîte visage pour couvrir toute la tête (cheveux inclus)
  const cx=bb.originX+bb.width/2,cy=bb.originY+bb.height/2,s=Math.max(bb.width,bb.height);
  const w=s*1.55,h=s*1.85;return {x:cx-w/2,y:cy-h*0.58,w,h};
}
function nms(boxes){boxes.sort((a,b)=>b.score-a.score);const out=[];
  for(const b of boxes){if(!out.some(o=>iou(o,b)>0.3))out.push(b);}return out;}
function updateTracks(boxes){
  const used=new Set();
  for(const t of S.tracks){
    let best=-1,bi=0;
    boxes.forEach((b,i)=>{if(used.has(i))return;const v=iou(t,b)+ (Math.hypot((t.x+t.w/2)-(b.x+b.w/2),(t.y+t.h/2)-(b.y+b.h/2))<t.w*0.6?0.2:0);if(v>bi){bi=v;best=i;}});
    if(best>=0&&bi>0.1){const b=boxes[best];used.add(best);
      const a=0.6; // lissage exponentiel
      const nx=t.x+(b.x-t.x)*a,ny=t.y+(b.y-t.y)*a;
      t.vx=nx-t.x;t.vy=ny-t.y;t.x=nx;t.y=ny;t.w+=(b.w-t.w)*a;t.h+=(b.h-t.h)*a;t.miss=0;t.hits++;
    }else{t.miss++;t.x+=t.vx*0.5;t.y+=t.vy*0.5;}
  }
  boxes.forEach((b,i)=>{if(!used.has(i))S.tracks.push({...b,vx:0,vy:0,miss:0,hits:1,id:S.nextId++});});
  S.tracks=S.tracks.filter(t=>t.miss<=8); // garde la boîte ~250 ms si la détection décroche
}

// ---- boucle ----
let lastVT=-1,fpsN=0,fpsT=performance.now(),detN=0,detBudget=0;
function detect(now){
  const which=S.range==='proche'?'short':S.range==='loin'?(S.dets.full?'full':'short'):(S.dets.full&&S.frame%3===2?'full':'short');
  const det=S.dets[which];if(!det)return;
  const t0=performance.now();
  let r;try{r=det.detectForVideo(video,now);}catch(e){console.warn(e);return;}
  S.detMs=S.detMs*0.9+(performance.now()-t0)*0.1;detN++;
  let boxes=r.detections.map(d=>({...headBox(d.boundingBox),score:d.categories[0]?.score||0}));
  if(which==='full'||S.range==='auto'){ // en mode auto, conserve les pistes de l'autre modèle
    boxes=nms(boxes);
  }
  updateTracks(boxes);
}
function render(){
  const vw=video.videoWidth,vh=video.videoHeight;if(!vw)return;
  if(canvas.width!==vw||canvas.height!==vh){canvas.width=vw;canvas.height=vh;}
  const mirror=S.facing==='user';
  ctx.save();if(mirror){ctx.translate(vw,0);ctx.scale(-1,1);}
  ctx.drawImage(video,0,0,vw,vh);
  ctx.fillStyle='#000';
  for(const t of S.tracks){ if(t.hits<1)continue;
    const pad=t.miss?0.08*t.w:0; // marge supplémentaire si la piste est extrapolée
    ctx.fillRect(t.x-pad,t.y-pad,t.w+2*pad,t.h+2*pad);}
  ctx.restore();
}
function loop(now,meta){
  const vt=meta?meta.mediaTime:video.currentTime;
  if(vt!==lastVT){
    lastVT=vt;S.frame++;
    // saute la détection si elle est trop lente pour garder la vidéo fluide
    const skip=S.detMs>28?2:1;
    if(S.frame%skip===0)detect(performance.now());
    render();fpsN++;
  }
  const n=performance.now();
  if(n-fpsT>1000){S.fps=fpsN*1000/(n-fpsT);S.detFps=detN*1000/(n-fpsT);fpsN=0;detN=0;fpsT=n;
    $('stats').textContent=`${S.fps.toFixed(0)} i/s · détection ${S.detMs.toFixed(0)} ms · ${S.delegate} · ${S.tracks.length} visage(s)`;}
  schedule();
}
function schedule(){
  if('requestVideoFrameCallback' in video)video.requestVideoFrameCallback(loop);
  else requestAnimationFrame(t=>loop(t,null));
}

// ---- capture ----
function canvasBlob(type,q){return new Promise(r=>canvas.toBlob(r,type,q));}
async function saveFile(blob,name){
  const file=new File([blob],name,{type:blob.type});
  if(navigator.canShare&&navigator.canShare({files:[file]})){
    try{await navigator.share({files:[file],title:'Caméra Masque'});return;}catch(e){if(e.name==='AbortError')return;}
  }
  const a=$('dl');a.href=URL.createObjectURL(blob);a.download=name;a.click();
  setTimeout(()=>URL.revokeObjectURL(a.href),30000);
  toast('Fichier téléchargé');
}
const stamp=()=>new Date().toISOString().replace(/[:.]/g,'-').slice(0,19);
async function takePhoto(){
  render(); // image courante avec carrés noirs
  canvas.classList.remove('flash');void canvas.offsetWidth;canvas.classList.add('flash');
  const blob=await canvasBlob('image/jpeg',0.92);
  await saveFile(blob,`masque-${stamp()}.jpg`);
}
function pickMime(){for(const m of['video/mp4;codecs=avc1','video/mp4','video/webm;codecs=vp9','video/webm'])if(window.MediaRecorder&&MediaRecorder.isTypeSupported(m))return m;return '';}
function toggleRec(){
  if(S.rec){S.rec.stop();return;}
  if(!window.MediaRecorder||!canvas.captureStream){toast("Enregistrement vidéo non pris en charge sur cet appareil");return;}
  const mime=pickMime();const chunks=[];
  const rec=new MediaRecorder(canvas.captureStream(30),mime?{mimeType:mime,videoBitsPerSecond:6e6}:undefined);
  rec.ondataavailable=e=>e.data.size&&chunks.push(e.data);
  rec.onstop=async()=>{S.rec=null;document.body.classList.remove('rec');
    const type=rec.mimeType||mime||'video/mp4';const blob=new Blob(chunks,{type});
    await saveFile(blob,`masque-${stamp()}.${type.includes('mp4')?'mp4':'webm'}`);};
  rec.start(1000);S.rec=rec;document.body.classList.add('rec');toast('Enregistrement…',1200);
}

// ---- UI ----
$('shutter').onclick=()=>S.videoMode?toggleRec():takePhoto();
$('modeBtn').onclick=()=>{if(S.rec)return;S.videoMode=!S.videoMode;document.body.classList.toggle('video',S.videoMode);$('modeBtn').textContent=S.videoMode?'Photo':'Vidéo';toast(S.videoMode?'Mode vidéo':'Mode photo',900);};
$('flipBtn').onclick=async()=>{if(S.rec)return;S.facing=S.facing==='user'?'environment':'user';try{await startCamera();}catch(e){toast('Caméra indisponible : '+e.message);}};
$('rangeBtn').onclick=()=>{S.range=RANGES[(RANGES.indexOf(S.range)+1)%3];$('rangeBtn').textContent='Portée : '+S.range;S.tracks=[];};

async function start(){
  const msg=$('msg');msg.textContent='';$('startBtn').disabled=true;$('startBtn').textContent='Chargement…';
  try{
    if(!navigator.mediaDevices?.getUserMedia)throw new Error("Caméra non accessible (HTTPS requis, ou ouvrez dans Safari).");
    await Promise.all([startCamera(),S.dets.short?null:initDetectors()]);
    $('start').hidden=true;schedule();
  }catch(e){
    console.error(e);
    msg.textContent=(e.name==='NotAllowedError')
      ?"Accès à la caméra refusé. Réglages › Safari › Caméra › Autoriser (ou Réglages › Caméra Masque), puis réessayez."
      :'Erreur : '+(e.message||e);
    $('startBtn').disabled=false;$('startBtn').textContent='Réessayer';
  }
}
$('startBtn').onclick=start;
document.addEventListener('visibilitychange',()=>{ // iOS coupe la caméra en arrière-plan
  if(!document.hidden&&S.stream&&S.stream.getVideoTracks().every(t=>t.readyState==='ended'))startCamera().catch(()=>{$('start').hidden=false;});
});
if('serviceWorker' in navigator)navigator.serviceWorker.register('sw.js');
if(params.get('autostart'))start();
