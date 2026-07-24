import {
  joinOnlineRoom,
  projectOnlineRoom,
  type OnlineRoomState,
} from "@/lib/online-game";
import {
  mutateStoredOnlineRoom,
  removeOnlineRoomMember,
  reserveOnlineRoomMember,
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
  let reserved:
    | { roomCode: string; playerId: string; nickname: string; token: string }
    | undefined;

  try {
    const code = await routeRoomCode(context);
    const body = await readJsonObject(request);
    reserved = await reserveOnlineRoomMember(code, body.nickname);
    const now = Date.now();
    const room = await mutateStoredOnlineRoom<OnlineRoomState>(
      reserved.roomCode,
      (state) =>
        joinOnlineRoom(
          state,
          { id: reserved!.playerId, name: reserved!.nickname },
          now,
        ),
    );
    const snapshot = projectOnlineRoom(room.state, reserved.playerId);

    return onlineJson(
      {
        roomCode: room.code,
        playerId: reserved.playerId,
        token: reserved.token,
        revision: room.revision,
        serverTime: Date.now(),
        snapshot,
      },
      { status: 201 },
    );
  } catch (error) {
    if (reserved) {
      try {
        await removeOnlineRoomMember(reserved.roomCode, reserved.playerId);
      } catch {
        // The reservation expires with its room; preserve the original error.
      }
    }
    return onlineApiErrorResponse(error);
  }
}
