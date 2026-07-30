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
  if (!waitingWorker && !(mobileWeb && installPrompt)) return null;

  return (
    <aside
      className={styles.prompt}
      role="status"
      aria-live="polite"
      aria-label={waitingWorker ? "새 버전 안내" : "앱 설치 안내"}
    >
      <span className={styles.crown} aria-hidden="true" />
      <span className={styles.copy}>
        <strong>{waitingWorker ? "새 버전이 준비됐어요" : "DALMUTI 설치"}</strong>
        <small>
          {waitingWorker
            ? "초기 화면에서 안전하게 업데이트할 수 있습니다."
            : "홈 화면에서 앱처럼 빠르게 실행하세요."}
        </small>
      </span>
      <button
        type="button"
        className={styles.action}
        disabled={installing || updating}
        onClick={waitingWorker ? applyUpdate : installApp}
      >
        {waitingWorker
          ? updating
            ? "적용 중"
            : "업데이트"
          : installing
            ? "설치 중"
            : "설치"}
      </button>
    </aside>
  );
}
