"use client";

import {
  createContext,
  useCallback,
  useContext,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import {
  APP_PREFERENCES_STORAGE_KEY,
  DEFAULT_APP_PREFERENCES,
  normalizeAppPreferences,
  parseAppPreferences,
  themeColor,
  type AppPreferences,
} from "@/lib/app-preferences";

type AppPreferencesContextValue = {
  preferences: AppPreferences;
  updatePreferences: (patch: Partial<AppPreferences>) => void;
};

const AppPreferencesContext = createContext<AppPreferencesContextValue | null>(
  null,
);

function applyPreferencesToDocument(preferences: AppPreferences): void {
  document.documentElement.dataset.theme = preferences.theme;
  document.documentElement.style.colorScheme = "dark";
  for (const meta of document.querySelectorAll<HTMLMetaElement>(
    'meta[name="theme-color"]',
  )) {
    meta.content = themeColor(preferences.theme);
  }
}

export function AppPreferencesProvider({ children }: { children: ReactNode }) {
  const [preferences, setPreferences] = useState<AppPreferences>({
    ...DEFAULT_APP_PREFERENCES,
  });
  const preferencesRef = useRef(preferences);

  const replacePreferences = useCallback(
    (next: AppPreferences, persist: boolean) => {
      preferencesRef.current = next;
      setPreferences(next);
      applyPreferencesToDocument(next);
      if (persist) {
        localStorage.setItem(
          APP_PREFERENCES_STORAGE_KEY,
          JSON.stringify(next),
        );
      }
    },
    [],
  );

  useLayoutEffect(() => {
    const stored = parseAppPreferences(
      localStorage.getItem(APP_PREFERENCES_STORAGE_KEY),
    );
    let active = true;
    queueMicrotask(() => {
      if (active) replacePreferences(stored, false);
    });

    const handleStorage = (event: StorageEvent) => {
      if (event.key !== APP_PREFERENCES_STORAGE_KEY) return;
      replacePreferences(parseAppPreferences(event.newValue), false);
    };
    window.addEventListener("storage", handleStorage);
    return () => {
      active = false;
      window.removeEventListener("storage", handleStorage);
    };
  }, [replacePreferences]);

  const updatePreferences = useCallback(
    (patch: Partial<AppPreferences>) => {
      replacePreferences(
        normalizeAppPreferences({ ...preferencesRef.current, ...patch }),
        true,
      );
    },
    [replacePreferences],
  );

  const value = useMemo(
    () => ({ preferences, updatePreferences }),
    [preferences, updatePreferences],
  );

  return (
    <AppPreferencesContext.Provider value={value}>
      {children}
    </AppPreferencesContext.Provider>
  );
}

export function useAppPreferences(): AppPreferencesContextValue {
  const value = useContext(AppPreferencesContext);
  if (!value) {
    throw new Error("useAppPreferences must be used inside AppPreferencesProvider");
  }
  return value;
}
