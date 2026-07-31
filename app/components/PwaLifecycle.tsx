"use client";

import { useCallback, useEffect, useState } from "react";
import styles from "./PwaLifecycle.module.css";

type InstallChoice = {
  outcome: "accepted" | "dismissed";
  platform: string;
};

interface BeforeInstallPromptEvent extends Event {
  prompt(): Promise<void>;
  userChoice: Promise<InstallChoice>;
}

type NavigatorWithUserAgentData = Navigator & {
  userAgentData?: {
    mobile?: boolean;
  };
};

function isStandalone() {
  if (typeof window === "undefined") return false;
  return (
    window.matchMedia("(display-mode: standalone)").matches ||
    ("standalone" in navigator &&
      Boolean((navigator as Navigator & { standalone?: boolean }).standalone))
  );
}

function isMobileWeb() {
  if (typeof window === "undefined") return false;
  const navigatorWithData = navigator as NavigatorWithUserAgentData;
  if (typeof navigatorWithData.userAgentData?.mobile === "boolean") {
    return navigatorWithData.userAgentData.mobile;
  }
  return (
    /Android|iPhone|iPad|iPod|Mobile/i.test(navigator.userAgent) ||
    (/Macintosh/i.test(navigator.userAgent) && navigator.maxTouchPoints > 1)
  );
}

function isIphoneWeb() {
  if (typeof window === "undefined") return false;
  return /iPhone|iPod/i.test(navigator.userAgent);
}

function isSafeInstallScreen() {
  const quickLanding = document.querySelector(".welcome-layer");
  const onlineEntry = document.querySelector('[class*="entryShell"]');
  const activeGame = document.querySelector(
    '[class*="gameShell"], [class*="lobbyShell"]',
  );

  return Boolean(quickLanding || (onlineEntry && !activeGame));
}

export function PwaLifecycle() {
  const [installPrompt, setInstallPrompt] =
    useState<BeforeInstallPromptEvent | null>(null);
  const [waitingWorker, setWaitingWorker] = useState<ServiceWorker | null>(null);
  const [safeScreen, setSafeScreen] = useState(false);
  const [mobileWeb] = useState(() => isMobileWeb());
  const [iphoneWeb] = useState(() => isIphoneWeb());
  const [iphoneGuideOpen, setIphoneGuideOpen] = useState(false);
  const [installing, setInstalling] = useState(false);
  const [updating, setUpdating] = useState(false);

  const refreshSafeScreen = useCallback(() => {
    setSafeScreen(isSafeInstallScreen());
  }, []);

  useEffect(() => {
    const initialCheck = window.requestAnimationFrame(refreshSafeScreen);
    const observer = new MutationObserver(refreshSafeScreen);
    observer.observe(document.body, {
      childList: true,
      subtree: true,
    });
    window.addEventListener("popstate", refreshSafeScreen);

    return () => {
      window.cancelAnimationFrame(initialCheck);
      observer.disconnect();
      window.removeEventListener("popstate", refreshSafeScreen);
    };
  }, [refreshSafeScreen]);

  useEffect(() => {
    const handleInstallPrompt = (event: Event) => {
      event.preventDefault();
      if (!isMobileWeb()) return;
      setInstallPrompt(event as BeforeInstallPromptEvent);
    };
    const handleInstalled = () => {
      setInstallPrompt(null);
      setInstalling(false);
    };

    window.addEventListener("beforeinstallprompt", handleInstallPrompt);
    window.addEventListener("appinstalled", handleInstalled);

    return () => {
      window.removeEventListener("beforeinstallprompt", handleInstallPrompt);
      window.removeEventListener("appinstalled", handleInstalled);
    };
  }, []);

  useEffect(() => {
    if (
      process.env.NODE_ENV !== "production" ||
      !window.isSecureContext ||
      !("serviceWorker" in navigator)
    ) {
      return;
    }

    let cancelled = false;
    let reloadForUpdate = false;
    let registration: ServiceWorkerRegistration | null = null;

    const inspectRegistration = (next: ServiceWorkerRegistration) => {
      registration = next;
      if (next.waiting) setWaitingWorker(next.waiting);

      next.addEventListener("updatefound", () => {
        const candidate = next.installing;
        if (!candidate) return;
        candidate.addEventListener("statechange", () => {
          if (
            candidate.state === "installed" &&
            navigator.serviceWorker.controller
          ) {
            setWaitingWorker(next.waiting ?? candidate);
          }
        });
      });
    };

    navigator.serviceWorker
      .register("/sw.js", { scope: "/", updateViaCache: "none" })
      .then((next) => {
        if (cancelled) return;
        inspectRegistration(next);
      })
      .catch(() => {
        // The game remains fully usable if service-worker registration fails.
      });

    const handleControllerChange = () => {
      if (!reloadForUpdate) return;
      reloadForUpdate = false;
      window.location.reload();
    };
    navigator.serviceWorker.addEventListener(
      "controllerchange",
      handleControllerChange,
    );

    const requestUpdate = () => {
      if (document.visibilityState !== "visible") return;
      registration?.update().catch(() => undefined);
    };
    document.addEventListener("visibilitychange", requestUpdate);
    const updateTimer = window.setInterval(requestUpdate, 60 * 60 * 1000);

    const handleApplyUpdate = () => {
      reloadForUpdate = true;
    };
    window.addEventListener("dalmuti:pwa-update-start", handleApplyUpdate);

    return () => {
      cancelled = true;
      window.clearInterval(updateTimer);
      document.removeEventListener("visibilitychange", requestUpdate);
      window.removeEventListener("dalmuti:pwa-update-start", handleApplyUpdate);
      navigator.serviceWorker.removeEventListener(
        "controllerchange",
        handleControllerChange,
      );
    };
  }, []);

  useEffect(() => {
    if (!iphoneGuideOpen) return;

    const previousOverflow = document.body.style.overflow;
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") setIphoneGuideOpen(false);
    };

    document.body.style.overflow = "hidden";
    window.addEventListener("keydown", closeOnEscape);

    return () => {
      document.body.style.overflow = previousOverflow;
      window.removeEventListener("keydown", closeOnEscape);
    };
  }, [iphoneGuideOpen]);

  const installApp = async () => {
    if (!installPrompt || installing) return;
    setInstalling(true);
    await installPrompt.prompt();
    const choice = await installPrompt.userChoice;
    setInstallPrompt(null);
    if (choice.outcome !== "accepted") setInstalling(false);
  };

  const applyUpdate = () => {
    if (!waitingWorker || updating) return;
    setUpdating(true);
    window.dispatchEvent(new Event("dalmuti:pwa-update-start"));
    waitingWorker.postMessage({ type: "SKIP_WAITING" });
  };

  if (!safeScreen || isStandalone()) return null;
  if (!waitingWorker && !(mobileWeb && installPrompt) && !iphoneWeb) return null;

  const showingIphoneInstall = !waitingWorker && iphoneWeb;

  return (
    <>
      <aside
        className={styles.prompt}
        role="status"
        aria-live="polite"
        aria-label={waitingWorker ? "새 버전 안내" : "앱 설치 안내"}
      >
        <span className={styles.crown} aria-hidden="true" />
        <span className={styles.copy}>
          <strong>
            {waitingWorker
              ? "새 버전이 준비됐어요"
              : showingIphoneInstall
                ? "iPhone에 DALMUTI 설치"
                : "DALMUTI 설치"}
          </strong>
          <small>
            {waitingWorker
              ? "초기 화면에서 안전하게 업데이트할 수 있습니다."
              : showingIphoneInstall
                ? "홈 화면에 추가해 앱처럼 실행하세요."
                : "홈 화면에서 앱처럼 빠르게 실행하세요."}
          </small>
        </span>
        <button
          type="button"
          className={styles.action}
          disabled={installing || updating}
          onClick={
            waitingWorker
              ? applyUpdate
              : showingIphoneInstall
                ? () => setIphoneGuideOpen(true)
                : installApp
          }
        >
          {waitingWorker
            ? updating
              ? "적용 중"
              : "업데이트"
            : showingIphoneInstall
              ? "설치 방법"
              : installing
                ? "설치 중"
                : "설치"}
        </button>
      </aside>

      {showingIphoneInstall && iphoneGuideOpen ? (
        <div
          className={styles.guideBackdrop}
          role="presentation"
          onClick={(event) => {
            if (event.target === event.currentTarget) {
              setIphoneGuideOpen(false);
            }
          }}
        >
          <section
            className={styles.guide}
            role="dialog"
            aria-modal="true"
            aria-labelledby="iphone-install-title"
            aria-describedby="iphone-install-description"
          >
            <button
              type="button"
              className={styles.guideClose}
              aria-label="iPhone 설치 안내 닫기"
              onClick={() => setIphoneGuideOpen(false)}
            >
              ×
            </button>

            <span className={styles.guideCrown} aria-hidden="true" />
            <span className={styles.guideEyebrow}>iPhone 앱 설치</span>
            <h2 id="iphone-install-title">DALMUTI를 홈 화면에 추가하세요</h2>
            <p id="iphone-install-description">
              별도 다운로드 없이 현재 웹사이트를 앱처럼 실행할 수 있습니다.
            </p>

            <ol className={styles.guideSteps}>
              <li>
                <span aria-hidden="true">1</span>
                <div>
                  <strong>브라우저의 공유 메뉴를 여세요</strong>
                  <small>Safari에서는 주소창의 공유 버튼을 누르세요.</small>
                </div>
              </li>
              <li>
                <span aria-hidden="true">2</span>
                <div>
                  <strong>홈 화면에 추가를 선택하세요</strong>
                  <small>
                    항목이 없다면 Safari에서 이 주소를 다시 열어주세요.
                  </small>
                </div>
              </li>
              <li>
                <span aria-hidden="true">3</span>
                <div>
                  <strong>웹 앱으로 열기를 켜고 추가하세요</strong>
                  <small>설치 후 DALMUTI 아이콘으로 바로 실행할 수 있습니다.</small>
                </div>
              </li>
            </ol>

            <button
              type="button"
              className={styles.guideConfirm}
              onClick={() => setIphoneGuideOpen(false)}
            >
              확인
            </button>
          </section>
        </div>
      ) : null}
    </>
  );
}
