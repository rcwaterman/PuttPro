// PuttPro Service Worker — minimal install-only SW for PWA prompt
const CACHE = 'puttproV3';

self.addEventListener('install', e => {
  self.skipWaiting();
  e.waitUntil(
    caches.open(CACHE).then(c =>
      c.addAll(['/mobile', '/static/manifest.json'])
       .catch(() => {}) // non-fatal if offline during install
    )
  );
});

self.addEventListener('activate', e => {
  e.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k)))
    )
  );
  self.clients.claim();
});

// Network-first: always try network, fall back to cache for navigation
self.addEventListener('fetch', e => {
  const url = new URL(e.request.url);

  // Pass through non-GET and API calls uncached
  if (e.request.method !== 'GET') return;
  if ([
    '/upload', '/status', '/stream/status', '/result', '/download',
    '/history', '/jobs', '/stats'
  ].some(p => url.pathname.startsWith(p))) return;

  e.respondWith(
    fetch(e.request)
      .then(res => {
        const clone = res.clone();
        caches.open(CACHE).then(c => c.put(e.request, clone));
        return res;
      })
      .catch(() => caches.match(e.request))
  );
});
