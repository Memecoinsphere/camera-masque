const V='camera-masque-v3';
const ASSETS=['./','index.html','style.css','app.js','vision_bundle.mjs','manifest.webmanifest','icon-180.png','icon-192.png','icon-512.png',
'models/blaze_face_short_range.tflite','models/blaze_face_full_range.tflite',
'wasm/vision_wasm_internal.js','wasm/vision_wasm_internal.wasm','wasm/vision_wasm_nosimd_internal.js','wasm/vision_wasm_nosimd_internal.wasm'];
self.addEventListener('install',e=>{e.waitUntil(caches.open(V).then(c=>c.addAll(ASSETS)).then(()=>self.skipWaiting()))});
self.addEventListener('activate',e=>{e.waitUntil(caches.keys().then(k=>Promise.all(k.filter(n=>n!==V).map(n=>caches.delete(n)))).then(()=>self.clients.claim()))});
self.addEventListener('fetch',e=>{
  if(e.request.method!=='GET')return;
  e.respondWith(caches.match(e.request,{ignoreSearch:true}).then(r=>r||fetch(e.request).then(res=>{
    if(res.ok&&new URL(e.request.url).origin===location.origin){const cl=res.clone();caches.open(V).then(c=>c.put(e.request,cl));}
    return res;}).catch(()=>caches.match('index.html'))));
});
