const V='camera-masque-v4';
const ASSETS=['./','index.html','style.css','app.js','vision_bundle.mjs','manifest.webmanifest','icon-180.png','icon-192.png','icon-512.png',
'models/blaze_face_short_range.tflite','models/blaze_face_full_range.tflite',
'wasm/vision_wasm_internal.js','wasm/vision_wasm_internal.wasm','wasm/vision_wasm_nosimd_internal.js','wasm/vision_wasm_nosimd_internal.wasm'];
self.addEventListener('install',e=>{self.skipWaiting();e.waitUntil(caches.open(V).then(c=>c.addAll(ASSETS.map(u=>new Request(u,{cache:'reload'})))));});
self.addEventListener('activate',e=>{e.waitUntil(caches.keys().then(k=>Promise.all(k.filter(n=>n!==V).map(n=>caches.delete(n)))).then(()=>self.clients.claim()));});
// réseau d'abord pour le code (index/app/style/sw/manifest), cache d'abord pour les gros fichiers (wasm, modèles)
const NET_FIRST=/(\/|index\.html|app\.js|style\.css|manifest\.webmanifest|vision_bundle\.mjs)$/;
self.addEventListener('fetch',e=>{
  const req=e.request;if(req.method!=='GET')return;const url=new URL(req.url);if(url.origin!==location.origin)return;
  const key=url.origin+url.pathname;
  if(req.mode==='navigate'||NET_FIRST.test(url.pathname)){
    e.respondWith(fetch(req,{cache:'no-store'}).then(res=>{if(res.ok){const c=res.clone();caches.open(V).then(ca=>ca.put(key,c));}return res;})
      .catch(()=>caches.match(key).then(r=>r||caches.match(new URL('index.html',self.registration.scope).href))));
  }else{
    e.respondWith(caches.match(key).then(r=>r||fetch(req).then(res=>{if(res.ok){const c=res.clone();caches.open(V).then(ca=>ca.put(key,c));}return res;})));
  }
});
