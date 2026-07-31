/**
 * Presentation values shared by quick match and online play.
 *
 * Keep network cadence (polling, command locks, reconnect grace) out of this
 * module. Those values are transport details rather than visible game timing.
 */
export const GAME_PRESENTATION_TIMING_MS = Object.freeze({
  rankCountdownStep: 1_100,
  rankIntro: 3_300,
  rankSelectionPause: 1_500,
  rankReveal: 3_400,
  rankConfirm: 2_600,
  revealIntro: 2_400,
  handReveal: 1_400,
  taxIntro: 2_400,
  taxStage: 6_000,
  playIntro: 2_600,
  revolutionIntro: 3_300,
  greatRevolutionSwap: 2_600,
  publicPlay: 2_250,
  publicPass: 1_500,
  dalmuti: 3_300,
  turn: 30_000,
  rankMove: 2_300,
  rankResultDelay: 280,
  publicCardMotion: 2_080,
  publicPassMotion: 1_380,
  taxCardMotion: 5_550,
  dalmutiFieldMotion: 3_250,
  dalmutiCardMotion: 3_100,
  dalmutiAutoPassMotion: 2_550,
  dalmutiAutoPassBanner: 3_050,
});

/**
 * Geometry used by the installed phone layout. Desktop/tablet layout can use
 * wider fans, but the installed app must render the same settled pile and
 * public action in quick and online modes.
 */
export const INSTALLED_MOBILE_PRESENTATION = Object.freeze({
  tableCardWidth: 88,
  tableCardHeight: 135,
  tableCardMaxStep: 32,
  tableCardSpread: 160,
  tableCardRotation: 1.1,
  tableCardLift: 0.9,
  tablePileMinHeight: 147,
  actionCardWidth: 92,
  actionCardHeight: 142,
  actionExpandedMaxStep: 54,
  actionExpandedSpread: 190,
  actionSettledMaxStep: 24,
  actionSettledSpread: 190,
  actionOriginSpread: 9,
  actionDelayMaxStep: 36,
  actionDelaySpread: 100,
  actionCaptionOffsetY: 101,
  dalmutiAutoPassOffset: 34,
  dalmutiAutoPassInitialDelay: 360,
  dalmutiAutoPassStagger: 90,
});

export function cappedPresentationStep(
  itemCount: number,
  maxStep: number,
  spread: number,
): number {
  if (itemCount <= 1) return 0;
  return Math.min(maxStep, spread / Math.max(1, itemCount - 1));
}

type PresentationTimingKey = keyof typeof GAME_PRESENTATION_TIMING_MS;

/**
 * Fails fast in development/tests if a mode silently drifts from a shared
 * visible timing. Keeping the local names makes the game code readable while
 * this guard keeps their values synchronized.
 */
export function assertPresentationTimingParity(
  mode: string,
  actual: Partial<Record<PresentationTimingKey, number>>,
): void {
  for (const key of Object.keys(actual) as PresentationTimingKey[]) {
    const expected = GAME_PRESENTATION_TIMING_MS[key];
    const received = actual[key];
    if (received !== expected) {
      throw new Error(
        `${mode} presentation timing mismatch for ${key}: ` +
          `expected ${expected}ms, received ${received}ms`,
      );
    }
  }
}
