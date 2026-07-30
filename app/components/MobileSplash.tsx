"use client";

import { useEffect, useState } from "react";
import styles from "./MobileSplash.module.css";

export function MobileSplash() {
  const [mounted, setMounted] = useState(true);
  const [leaving, setLeaving] = useState(false);

  useEffect(() => {
    const leaveTimer = window.setTimeout(() => setLeaving(true), 1_250);
    const removeTimer = window.setTimeout(() => setMounted(false), 1_700);

    return () => {
      window.clearTimeout(leaveTimer);
      window.clearTimeout(removeTimer);
    };
  }, []);

  if (!mounted) return null;

  return (
    <div
      className={`${styles.splash} ${leaving ? styles.leaving : ""}`}
      aria-hidden="true"
    >
      <span className={styles.halo} />
      <span className={styles.crown} />
      <strong>DALMUTI</strong>
      <small>THE GREAT DALMUTI</small>
    </div>
  );
}
