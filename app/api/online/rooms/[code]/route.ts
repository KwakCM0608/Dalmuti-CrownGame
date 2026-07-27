import {
  advanceOnlineRoom,
  projectOnlineRoom,
  type OnlineRoomState,
} from "@/lib/online-game";
import {
  authenticateOnlineRoomRequest,
  mutateStoredOnlineRoom,
  readOnlineRoomChatMessages,
  readOnlineRoomEmotes,
} from "@/lib/online-room-store";
import {
  onlineApiErrorResponse,
  onlineJson,
  optionalChatSequence,
  optionalEventSequence,
  routeRoomCode,
} from "../../_shared";

interface RouteContext {
  params: Promise<{ code: string }> | { code: string };
}

export async function GET(
  request: Request,
  context: RouteContext,
): Promise<Response> {
  try {
    const code = await routeRoomCode(context);
    const member = await authenticateOnlineRoomRequest(request, code);
    const sinceEventSeq = optionalEventSequence(request);
    const sinceChatSeq = optionalChatSequence(request);
    const now = Date.now();
    const room = await mutateStoredOnlineRoom<OnlineRoomState>(
      code,
      (state) => advanceOnlineRoom(state, now),
    );
    const snapshot = projectOnlineRoom(
      room.state,
      member.playerId,
      sinceEventSeq,
    );
    const [chat, emotes] = await Promise.all([
      readOnlineRoomChatMessages(code, sinceChatSeq),
      readOnlineRoomEmotes(code, now),
    ]);

    return onlineJson({
      roomCode: room.code,
      playerId: member.playerId,
      revision: room.revision,
      serverTime: Date.now(),
      snapshot,
      chatMessages: chat.messages,
      latestChatSeq: chat.latestSequence,
      emotes,
    });
  } catch (error) {
    return onlineApiErrorResponse(error);
  }
}
