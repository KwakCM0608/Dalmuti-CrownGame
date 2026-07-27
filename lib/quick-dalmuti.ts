type QuickDalmutiPlayer = {
  id: string;
};

type QuickDalmutiHands = Readonly<Record<string, readonly unknown[]>>;

export type QuickDalmutiAutoPassResolution = {
  autoPassedPlayerIds: string[];
  nextPlayerIndex: number;
};

export function resolveQuickDalmutiAutoPass<
  TPlayer extends QuickDalmutiPlayer,
>(
  players: readonly TPlayer[],
  hands: QuickDalmutiHands,
  actorId: string,
  actorIndex: number,
): QuickDalmutiAutoPassResolution {
  const autoPassedPlayerIds = players
    .filter(
      (player) =>
        player.id !== actorId && (hands[player.id]?.length ?? 0) > 0,
    )
    .map((player) => player.id);

  if ((hands[actorId]?.length ?? 0) > 0) {
    return {
      autoPassedPlayerIds,
      nextPlayerIndex: actorIndex,
    };
  }

  for (let step = 1; step <= players.length; step += 1) {
    const index = (actorIndex + step) % players.length;
    if ((hands[players[index].id]?.length ?? 0) > 0) {
      return {
        autoPassedPlayerIds,
        nextPlayerIndex: index,
      };
    }
  }

  return {
    autoPassedPlayerIds,
    nextPlayerIndex: actorIndex,
  };
}
