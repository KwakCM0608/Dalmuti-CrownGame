const CACHE_PREFIX = "dalmuti-pwa";
const CACHE_VERSION = "2026-08-14-halloween-dalmuti-name-v22";
const PRECACHE = `${CACHE_PREFIX}-precache-${CACHE_VERSION}`;
const RUNTIME = `${CACHE_PREFIX}-runtime-${CACHE_VERSION}`;
const LEGACY_AUTO_UPDATE_PRECACHES = [
  `${CACHE_PREFIX}-precache-2026-08-14-auto-pass-icon-v19`,
  `${CACHE_PREFIX}-precache-2026-08-14-installed-update-prompt-v20`,
];
const OFFLINE_URL = "/offline.html";

const PRECACHE_URLS = [
  OFFLINE_URL,
  "/brand-dalmuti-crown.png",
  "/pwa/icon-192.png",
  "/pwa/icon-512.png",
  "/pwa/icon-maskable-512.png",
  "/pwa/apple-touch-icon.png",
  "/pwa/icon-v3-192.png",
  "/pwa/icon-v3-512.png",
  "/pwa/icon-v3-1024.png",
  "/pwa/icon-maskable-v3-512.png",
  "/pwa/apple-touch-icon-v3.png",
  "/cards/back.webp",
  "/cards/halloween/back.webp",
  "/themes/halloween/crown.webp",
  "/themes/halloween/dalmuti-hand-field-atlas-v2.png",
  "/themes/halloween/ink-wash-field-texture-v2.webp",
  "/themes/halloween/ink-impact-bloom-mask-v1.png",
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(PRECACHE).then(async (cache) => {
      await Promise.allSettled(
        PRECACHE_URLS.map(async (url) => {
          const response = await fetch(url, { cache: "reload" });
          if (response.ok) await cache.put(url, response);
        }),
      );
      const existingCaches = await caches.keys();
      if (
        LEGACY_AUTO_UPDATE_PRECACHES.some((cacheName) =>
          existingCaches.includes(cacheName),
        )
      ) {
        await self.skipWaiting();
      }
    }),
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) =>
        Promise.all(
          keys
            .filter(
              (key) =>
                key.startsWith(`${CACHE_PREFIX}-`) &&
                key !== PRECACHE &&
                key !== RUNTIME,
            )
            .map((key) => caches.delete(key)),
        ),
      )
      .then(() => self.clients.claim()),
  );
});

function isOnlineApi(pathname) {
  return pathname === "/api/online" || pathname.startsWith("/api/online/");
}

function isCardOrBrandAsset(pathname) {
  return (
    pathname.startsWith("/cards/") ||
    pathname.startsWith("/pwa/") ||
    pathname.startsWith("/themes/") ||
    pathname === "/brand-dalmuti-crown.png"
  );
}

function isBuildAsset(pathname) {
  return /\.(?:css|js|woff2?|webp|png|jpg|jpeg|gif)$/i.test(pathname);
}

async function cacheFirst(request) {
  const cached = await caches.match(request);
  if (cached) return cached;

  const response = await fetch(request);
  if (response.ok) {
    const cache = await caches.open(RUNTIME);
    await cache.put(request, response.clone());
  }
  return response;
}

async function staleWhileRevalidate(request) {
  const cache = await caches.open(RUNTIME);
  const cached = await cache.match(request);
  const network = fetch(request)
    .then(async (response) => {
      if (response.ok) await cache.put(request, response.clone());
      return response;
    })
    .catch(() => null);

  return cached ?? (await network) ?? Response.error();
}

async function networkNavigation(request) {
  try {
    return await fetch(request);
  } catch {
    return (
      (await caches.match(OFFLINE_URL)) ??
      new Response("오프라인 상태입니다.", {
        status: 503,
        headers: { "Content-Type": "text/plain; charset=utf-8" },
      })
    );
  }
}

self.addEventListener("fetch", (event) => {
  const request = event.request;

  // Commands and private online snapshots always go directly to the server.
  // They must never be written to an offline cache.
  if (request.method !== "GET") return;

  const url = new URL(request.url);
  if (url.origin !== self.location.origin) return;
  if (isOnlineApi(url.pathname)) {
    event.respondWith(fetch(request));
    return;
  }

  if (request.mode === "navigate") {
    event.respondWith(networkNavigation(request));
    return;
  }

  if (isCardOrBrandAsset(url.pathname)) {
    event.respondWith(cacheFirst(request));
    return;
  }

  if (isBuildAsset(url.pathname)) {
    event.respondWith(staleWhileRevalidate(request));
  }
});

self.addEventListener("message", (event) => {
  if (event.data?.type === "SKIP_WAITING") {
    self.skipWaiting();
  }
});
