const INSTALLED_SPLASH_SESSION_KEY = "dalmuti-installed-splash-v2";
const INSTALLED_SPLASH_DATA_ATTRIBUTE = "data-dalmuti-pwa-splash";
const INSTALLED_SPLASH_CLEANUP_MS = 1_400;

const bootstrapInstalledSplash = `(() => {
  const root = document.documentElement;
  const installed =
    window.matchMedia("(display-mode: standalone)").matches ||
    window.matchMedia("(display-mode: fullscreen)").matches ||
    Boolean(navigator.standalone);
  const mobile =
    Boolean(navigator.userAgentData && navigator.userAgentData.mobile) ||
    /Android|iPhone|iPad|iPod|Mobile/i.test(navigator.userAgent);
  const reducedMotion = window.matchMedia(
    "(prefers-reduced-motion: reduce)"
  ).matches;
  const androidTwa =
    new URLSearchParams(window.location.search).get("source") ===
    "android-twa";
  let firstLaunchInSession = true;

  try {
    firstLaunchInSession =
      sessionStorage.getItem("${INSTALLED_SPLASH_SESSION_KEY}") !== "shown";
    if (firstLaunchInSession) {
      sessionStorage.setItem("${INSTALLED_SPLASH_SESSION_KEY}", "shown");
    }
  } catch {
    firstLaunchInSession = true;
  }

  const show =
    installed &&
    mobile &&
    !androidTwa &&
    !reducedMotion &&
    firstLaunchInSession;
  root.setAttribute(
    "${INSTALLED_SPLASH_DATA_ATTRIBUTE}",
    show ? "show" : "skip"
  );
  if (show) {
    window.setTimeout(() => {
      root.setAttribute("${INSTALLED_SPLASH_DATA_ATTRIBUTE}", "done");
    }, ${INSTALLED_SPLASH_CLEANUP_MS});
  }
})();`;

export function InstalledPwaSplash() {
  return (
    <>
      <script
        dangerouslySetInnerHTML={{ __html: bootstrapInstalledSplash }}
      />
      <div className="installed-pwa-splash" aria-hidden="true">
        <span className="installed-pwa-splash-glow" />
        <span className="installed-pwa-splash-art" />
      </div>
    </>
  );
}
