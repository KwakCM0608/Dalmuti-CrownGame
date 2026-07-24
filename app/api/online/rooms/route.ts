import {
  createOnlineRoom,
  projectOnlineRoom,
  type OnlineRoomState,
} from "@/lib/online-game";
import { createStoredOnlineRoom } from "@/lib/online-room-store";
import {
  onlineApiErrorResponse,
  onlineJson,
  readJsonObject,
} from "../_shared";

export async function POST(request: Request): Promise<Response> {
  try {
    const body = await readJsonObject(request);
    const now = Date.now();
    const { room, session } = await createStoredOnlineRoom<OnlineRoomState>(
      body.nickname,
      ({ roomCode, playerId, nickname }) =>
        createOnlineRoom(
          roomCode,
          { id: playerId, name: nickname },
          now,
        ),
    );
    const snapshot = projectOnlineRoom(room.state, session.playerId);

    return onlineJson(
      {
        roomCode: room.code,
        playerId: session.playerId,
        token: session.token,
        revision: room.revision,
        serverTime: Date.now(),
        snapshot,
      },
      { status: 201 },
    );
  } catch (error) {
    return onlineApiErrorResponse(error);
  }
}
