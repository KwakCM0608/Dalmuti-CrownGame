import type { BotDifficulty } from "../bot-strategy.ts";

export type OnlineRoomPhase =
  | "lobby"
  | "rank-intro"
  | "rank-selection"
  | "rank-reveal"
  | "rank-confirm"
  | "reveal-intro"
  | "hand-reveal"
  | "revolution"
  | "revolution-intro"
  | "great-revolution-swap"
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
  isBot: boolean;
  botDifficulty: BotDifficulty | null;
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

export type OnlineRankCardState = {
  slotIndex: number;
  rank: number;
  claimedByPlayerId: string | null;
  claimedAt: number | null;
};

export type OnlineRankSelectionState = {
  cards: OnlineRankCardState[];
  introStartedAt: number;
  countdownStartsAt: number;
  countdownEndsAt: number;
  revealAt: number | null;
  revealEndsAt: number | null;
};

export type OnlineDeclaredRevolution = {
  round: number;
  playerId: string;
  kind: "revolution" | "great-revolution";
};

export type OnlineEventVisibility = "public" | "private";

export type OnlineEventType =
  | "ROOM_CREATED"
  | "PLAYER_JOINED"
  | "BOT_ADDED"
  | "BOT_REMOVED"
  | "PLAYER_READY_CHANGED"
  | "RANK_CHOICE_INTRO_STARTED"
  | "RANK_CHOICE_STARTED"
  | "RANK_CARD_CHOSEN"
  | "RANK_CHOICES_LOCKED"
  | "RANK_CARDS_REVEALED"
  | "RANK_ORDER_ASSIGNED"
  | "RANK_CONFIRM_STARTED"
  | "DEAL_SEALED"
  | "MATCH_STARTED"
  | "HAND_REVEAL_STARTED"
  | "HAND_REVEALED"
  | "REVOLUTION_DECISION_STARTED"
  | "REVOLUTION_DECLARED"
  | "REVOLUTION_DECLINED"
  | "REVOLUTION_INTRO_STARTED"
  | "GREAT_REVOLUTION_RANK_SWAP_STARTED"
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
  | "DALMUTI_EFFECT"
  | "PLAYER_PASSED"
  | "TRICK_CLEARED"
  | "PLAYER_FINISHED"
  | "ROUND_ENDED"
  | "PLAYER_LEFT"
  | "ROOM_RESET";

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
      type: "ADD_BOT";
      difficulty?: BotDifficulty;
    })
  | (OnlineCommandBase & {
      type: "REMOVE_BOT";
      botId: string;
    })
  | (OnlineCommandBase & {
      type: "START_MATCH";
    })
  | (OnlineCommandBase & {
      type: "CHOOSE_RANK_CARD";
      slotIndex: number;
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
    })
  | (OnlineCommandBase & {
      type: "RESET_ROOM";
    })
  | (OnlineCommandBase & {
      type: "LEAVE_ROOM";
    });

export type OnlinePhaseDurations = {
  rankChoiceIntroMs: number;
  rankRevealDelayMs: number;
  rankRevealMs: number;
  rankConfirmMs: number;
  revealIntroMs: number;
  handRevealMs: number;
  revolutionDecisionMs: number;
  revolutionIntroMs: number;
  greatRevolutionSwapMs: number;
  taxIntroMs: number;
  taxSelectionMs: number;
  taxTributeMs: number;
  taxReturnMs: number;
  playIntroMs: number;
};

export type OnlineEngineDeps = {
  randomInt?: (maxExclusive: number) => number;
  durations?: Partial<OnlinePhaseDurations>;
  temporaryGreatRevolutionTestMode?: boolean;
};

export type OnlineRoomState = {
  code: string;
  revision: number;
  phase: OnlineRoomPhase;
  phaseEndsAt: number | null;
  turnDeadline: number | null;
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
  rankSelection: OnlineRankSelectionState | null;
  revolutionHolderId: string | null;
  declaredRevolution: OnlineDeclaredRevolution | null;
  taxExchanges: OnlineTaxExchange[];
  actionLockUntil: number | null;
  botActionAt: number | null;
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
  isBot: boolean;
  botDifficulty: BotDifficulty | null;
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
  turnDeadline: number | null;
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
  rankSelection: {
    stage: "intro" | "selecting" | "locked" | "revealed" | "confirmed";
    cards: Array<{
      slotIndex: number;
      claimedByPlayerId: string | null;
      revealedRank: number | null;
    }>;
    introStartedAt: number;
    countdownStartsAt: number;
    countdownEndsAt: number;
    revealAt: number | null;
    revealEndsAt: number | null;
    canChoose: boolean;
    selectedSlotIndex: number | null;
  } | null;
  declaredRevolution: OnlineDeclaredRevolution | null;
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
