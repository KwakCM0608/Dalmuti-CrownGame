import {
  advanceOnlineRoom,
  applyOnlineCommand,
  projectOnlineRoom,
  type OnlineCommand,
  type OnlineRoomState,
} from "@/lib/online-game";
import {
  OnlineStoreError,
  authenticateOnlineRoomRequest,
  mutateStoredOnlineRoom,
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
    const body = await readJsonObject(request);
    const candidate = body.command ?? body;
    if (!candidate || typeof candidate !== "object" || Array.isArray(candidate)) {
      throw new OnlineStoreError(
        "INVALID_COMMAND",
        "게임 명령 형식을 확인해 주세요.",
        400,
      );
    }
    const command = candidate as OnlineCommand;
    const now = Date.now();

    const room = await mutateStoredOnlineRoom<OnlineRoomState>(
      code,
      (state) => {
        const advanced = advanceOnlineRoom(state, now);
        return applyOnlineCommand(
          advanced,
          member.playerId,
          command,
          now,
        );
      },
    );
    const snapshot = projectOnlineRoom(room.state, member.playerId);

    return onlineJson({
      roomCode: room.code,
      playerId: member.playerId,
      revision: room.revision,
      serverTime: Date.now(),
      snapshot,
    });
  } catch (error) {
    return onlineApiErrorResponse(error);
  }
}
