"use client";

import { useCallback, useSyncExternalStore } from "react";
import { AUTO_PASS_STORAGE_KEY } from "@/lib/auto-pass";

const preferenceListeners = new Set<() => void>();
let sessionPreference = true;

function subscribeToAutoPassPreference(listener: () => void) {
  preferenceListeners.add(listener);
  const handleStorage = (event: StorageEvent) => {
    if (event.key === AUTO_PASS_STORAGE_KEY) listener();
  };
  window.addEventListener("storage", handleStorage);
  return () => {
    preferenceListeners.delete(listener);
    window.removeEventListener("storage", handleStorage);
  };
}

function getAutoPassPreference() {
  try {
    const storedPreference = window.localStorage.getItem(AUTO_PASS_STORAGE_KEY);
    return storedPreference === null
      ? sessionPreference
      : storedPreference !== "false";
  } catch {
    return sessionPreference;
  }
}

export function useAutoPassPreference() {
  const autoPassEnabled = useSyncExternalStore(
    subscribeToAutoPassPreference,
    getAutoPassPreference,
    () => true,
  );

  const setAutoPassEnabled = useCallback((enabled: boolean) => {
    sessionPreference = enabled;
    try {
      window.localStorage.setItem(AUTO_PASS_STORAGE_KEY, String(enabled));
    } catch {
      // The switch still works for the current session without persistence.
    }
    preferenceListeners.forEach((listener) => listener());
  }, []);

  return { autoPassEnabled, setAutoPassEnabled };
}
