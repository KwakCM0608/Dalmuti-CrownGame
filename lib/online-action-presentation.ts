export const REMOTE_ACTION_REPLAY_WINDOW_MS = 3_500;

export type OnlineActionPresentationEvent = {
  id: string;
  seq: number;
  type: string;
  startsAt: number;
  durationMs: number;
  actorPlayerId: string | null;
  data: Record<string, unknown>;
};

const TABLE_ACTION_EVENT_TYPES = new Set([
  "CARDS_PLAYED",
  "DALMUTI_EFFECT",
  "PLAYER_PASSED",
]);

export function isOnlineTableActionEvent(
  event: Pick<OnlineActionPresentationEvent, "type">,
): boolean {
  return TABLE_ACTION_EVENT_TYPES.has(event.type);
}

function actionBatchKey(
  event: Pick<
    OnlineActionPresentationEvent,
    "actorPlayerId" | "startsAt"
  >,
): string {
  return `${event.startsAt}:${event.actorPlayerId ?? ""}`;
}

/**
 * Selects public opponent actions that still belong to the live presentation
 * timeline. The server emits CARDS_PLAYED, DALMUTI_EFFECT and several automatic
 * PLAYER_PASSED events for one Dalmuti play; those are collapsed into the one
 * DALMUTI_EFFECT animation the UI already knows how to render.
 */
export function collectRemoteActionPresentations<
  T extends OnlineActionPresentationEvent,
>(
  events: readonly T[],
  viewerId: string,
  serverTime: number,
  seenIds: ReadonlySet<string>,
): { events: T[]; seenIds: string[] } {
  const remoteActions = events
    .filter(
      (event) =>
        isOnlineTableActionEvent(event) &&
        event.actorPlayerId !== null &&
        event.actorPlayerId !== viewerId &&
        !seenIds.has(event.id),
    )
    .sort((left, right) => left.seq - right.seq);
  const newlySeenIds = remoteActions.map((event) => event.id);
  const dalmutiBatches = new Set(
    remoteActions
      .filter((event) => event.type === "DALMUTI_EFFECT")
      .map(actionBatchKey),
  );

  return {
    events: remoteActions.filter((event) => {
      const age = Math.max(0, serverTime - event.startsAt);
      if (age > REMOTE_ACTION_REPLAY_WINDOW_MS) return false;
      if (
        event.type === "CARDS_PLAYED" &&
        dalmutiBatches.has(actionBatchKey(event))
      ) {
        return false;
      }
      if (
        event.type === "PLAYER_PASSED" &&
        event.data.reason === "dalmuti"
      ) {
        return false;
      }
      return true;
    }),
    seenIds: newlySeenIds,
  };
}
