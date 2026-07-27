import {
  advanceOnlineRoom,
  applyOnlineCommand,
  type OnlineCommand,
  type OnlineRoomState,
} from "@/lib/online-game";
import {
  authenticateOnlineRoomRequest,
  mutateStoredOnlineRoom,
  removeOnlineRoomMember,
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
      type: "LEAVE_ROOM",
    };
    const now = Date.now();
    const room = await mutateStoredOnlineRoom<OnlineRoomState>(
      code,
      (state) =>
        applyOnlineCommand(
          advanceOnlineRoom(state, now),
          member.playerId,
          command,
          now,
        ),
    );
    await removeOnlineRoomMember(code, member.playerId);

    return onlineJson({
      roomCode: room.code,
      playerId: member.playerId,
      revision: room.revision,
      serverTime: Date.now(),
      left: true,
    });
  } catch (error) {
    return onlineApiErrorResponse(error);
  }
}
