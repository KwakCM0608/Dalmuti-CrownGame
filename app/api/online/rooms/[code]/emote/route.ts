import type { OnlineRoomState } from "@/lib/online-game";
import {
  OnlineStoreError,
  appendOnlineRoomEmote,
  authenticateOnlineRoomRequest,
  readStoredOnlineRoom,
} from "@/lib/online-room-store";
import {
  onlineApiErrorResponse,
  onlineJson,
  readJsonObject,
  routeRoomCode,
} from "../../../_shared";

interface RouteContext {
  params: Promise<{ code: string }> | { code: string };
}

export async function POST(
  request: Request,
  context: RouteContext,
): Promise<Response> {
  try {
    const code = await routeRoomCode(context);
    const member = await authenticateOnlineRoomRequest(request, code);
    const room = await readStoredOnlineRoom<OnlineRoomState>(code);
    if (
      !room ||
      !room.state.players.some((player) => player.id === member.playerId)
    ) {
      throw new OnlineStoreError(
        "PLAYER_NOT_FOUND",
        "이 방에 다시 참가해 주세요.",
        404,
      );
    }
    const body = await readJsonObject(request);
    const emote = await appendOnlineRoomEmote(
      code,
      member,
      body.id ?? body.requestId,
      body.emoteId,
    );

    return onlineJson({
      roomCode: code,
      playerId: member.playerId,
      emote,
      serverTime: Date.now(),
    });
  } catch (error) {
    return onlineApiErrorResponse(error);
  }
}
