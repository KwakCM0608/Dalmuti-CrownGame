export type OnlineRoomPhase =
  | "lobby"
  | "reveal-intro"
  | "hand-reveal"
  | "revolution"
  | "tax-intro"
  | "tax-selection"
  | "tax-tribute"
  | "tax-return"
  | "play-intro"
  | "playing"
  | "round-end";

export type OnlineRole =
  | "great-dalmuti"
  | "lesser-dalmuti"
  | "merchant"
  | "lesser-peon"
  | "great-peon";

export type OnlineCard = {
  id: string;
  rank: number;
};

export type OnlinePlayerInput = {
  id: string;
  name: string;
  monogram?: string;
};

export type OnlinePlayerState = {
  id: string;
  name: string;
  monogram: string;
  role: OnlineRole;
  ready: boolean;
  connected: boolean;
  joinedAt: number;
  score: number;
};

export type OnlineTable = {
  rank: number;
  count: number;
  playerId: string;
  cards: OnlineCard[];
};

export type OnlineTaxExchange = {
  nobleId: string;
  peonId: string;
  count: number;
  peonCardIds: string[];
  nobleCardIds: string[] | null;
};

export type OnlineEventVisibility = "public" | "private";

export type OnlineEventType =
  | "ROOM_CREATED"
  | "PLAYER_JOINED"
  | "PLAYER_READY_CHANGED"
  | "DEAL_SEALED"
  | "MATCH_STARTED"
  | "HAND_REVEAL_STARTED"
  | "HAND_REVEALED"
  | "REVOLUTION_DECISION_STARTED"
  | "REVOLUTION_DECLARED"
  | "REVOLUTION_DECLINED"
  | "TAX_INTRO_STARTED"
  | "TAX_SELECTION_STARTED"
  | "TAX_RETURN_SELECTED"
  | "TAX_TRIBUTE_STARTED"
  | "TAX_TRIBUTE"
  | "TAX_RETURN_STARTED"
  | "TAX_RETURN"
  | "PLAY_INTRO_STARTED"
  | "TURN_STARTED"
  | "CARDS_PLAYED"
  | "PLAYER_PASSED"
  | "TRICK_CLEARED"
  | "PLAYER_FINISHED"
  | "ROUND_ENDED";

export type OnlineEvent = {
  seq: number;
  type: OnlineEventType;
  at: number;
  visibility: OnlineEventVisibility;
  playerIds?: string[];
  payload: Record<string, unknown>;
};

type OnlineCommandBase = {
  id: string;
  expectedRevision?: number;
};

export type OnlineCommand =
  | (OnlineCommandBase & {
      type: "SET_READY";
      ready: boolean;
    })
  | (OnlineCommandBase & {
      type: "START_MATCH";
    })
  | (OnlineCommandBase & {
      type: "CHOOSE_REVOLUTION";
      declare: boolean;
    })
  | (OnlineCommandBase & {
      type: "SELECT_TAX_RETURN";
      cardIds: string[];
    })
  | (OnlineCommandBase & {
      type: "PLAY_CARDS";
      cardIds: string[];
    })
  | (OnlineCommandBase & {
      type: "PASS";
    })
  | (OnlineCommandBase & {
      type: "START_NEXT_ROUND";
    });

export type OnlinePhaseDurations = {
  revealIntroMs: number;
  handRevealMs: number;
  revolutionDecisionMs: number;
  taxIntroMs: number;
  taxSelectionMs: number;
  taxTributeMs: number;
  taxReturnMs: number;
  playIntroMs: number;
};

export type OnlineEngineDeps = {
  randomInt?: (maxExclusive: number) => number;
  durations?: Partial<OnlinePhaseDurations>;
};

export type OnlineRoomState = {
  code: string;
  revision: number;
  phase: OnlineRoomPhase;
  phaseEndsAt: number | null;
  round: number;
  hostId: string;
  players: OnlinePlayerState[];
  hands: Record<string, OnlineCard[]>;
  dealSealed: boolean;
  currentIndex: number;
  table: OnlineTable | null;
  lastPlayedId: string | null;
  passedPlayerIds: string[];
  finishOrder: string[];
  revolutionHolderId: string | null;
  taxExchanges: OnlineTaxExchange[];
  actionLockUntil: number | null;
  events: OnlineEvent[];
  nextEventSeq: number;
  processedCommandIds: string[];
  durations: OnlinePhaseDurations;
  createdAt: number;
  updatedAt: number;
};

export type OnlineSnapshotPlayer = {
  id: string;
  name: string;
  monogram: string;
  role: OnlineRole;
  ready: boolean;
  connected: boolean;
  handCount: number;
  finishedPlace: number | null;
  score: number;
};

export type OnlineSnapshot = {
  code: string;
  revision: number;
  phase: OnlineRoomPhase;
  phaseEndsAt: number | null;
  round: number;
  viewerId: string;
  hostId: string;
  dealSealed: boolean;
  minPlayers: 4;
  maxPlayers: 8;
  players: OnlineSnapshotPlayer[];
  hand: OnlineCard[] | null;
  table: OnlineTable | null;
  currentPlayerId: string | null;
  lastPlayedId: string | null;
  actionLockUntil: number | null;
  passedPlayerIds: string[];
  finishOrder: string[];
  events: OnlineEvent[];
  latestEventSeq: number;
  tax: {
    requiredReturnCount: number;
    selectedReturnCount: number;
    waitingForPlayerIds: string[];
  } | null;
  revolution: {
    holderId: string | null;
    canChoose: boolean;
  } | null;
};
