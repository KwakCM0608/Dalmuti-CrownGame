import {
  advanceOnlineRoom,
  applyOnlineCommand,
  type OnlineCommand,
  type OnlineRoomState,
} from "@/lib/online-game";
import {
  OnlineStoreError,
  authenticateOnlineRoomRequest,
  deleteStoredOnlineRoom,
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
    const body = await readJsonObject(request);
    const command: OnlineCommand = {
      id: String(body.id ?? body.commandId ?? ""),
      ...(typeof body.expectedRevision === "number"
        ? { expectedRevision: body.expectedRevision }
        : {}),
      type: "RESET_ROOM",
    };
    const now = Date.now();
    const room = await readStoredOnlineRoom<OnlineRoomState>(code);
    if (!room) {
      throw new OnlineStoreError(
        "ROOM_NOT_FOUND",
        "The room no longer exists.",
        404,
      );
    }
    applyOnlineCommand(
      advanceOnlineRoom(room.state, now),
      member.playerId,
      command,
      now,
    );
    await deleteStoredOnlineRoom(code);

    return onlineJson({
      roomCode: code,
      playerId: member.playerId,
      serverTime: Date.now(),
      reset: true,
    });
  } catch (error) {
    return onlineApiErrorResponse(error);
  }
}
