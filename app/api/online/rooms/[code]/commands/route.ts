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
  deleteStoredOnlineRoom,
  mutateStoredOnlineRoom,
  readStoredOnlineRoom,
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

    if (command.type === "RESET_ROOM") {
      const stored = await readStoredOnlineRoom<OnlineRoomState>(code);
      if (!stored) {
        throw new OnlineStoreError(
          "ROOM_NOT_FOUND",
          "The room no longer exists.",
          404,
        );
      }
      // Validate membership, host authority, command id, and revision through
      // the same server-authoritative engine before performing the deletion.
      applyOnlineCommand(
        advanceOnlineRoom(stored.state, now),
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
    }

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
    if (command.type === "LEAVE_ROOM") {
      await removeOnlineRoomMember(code, member.playerId);
      return onlineJson({
        roomCode: room.code,
        playerId: member.playerId,
        revision: room.revision,
        serverTime: Date.now(),
        left: true,
      });
    }
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
