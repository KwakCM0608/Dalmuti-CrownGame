import { OnlineGameError } from "@/lib/online-game";
import {
  OnlineStoreError,
  onlineStoreErrorResponse,
} from "@/lib/online-room-store";

const MAX_JSON_BYTES = 24 * 1_024;

export async function readJsonObject(
  request: Request,
): Promise<Record<string, unknown>> {
  const contentLength = Number(request.headers.get("content-length") ?? 0);
  if (Number.isFinite(contentLength) && contentLength > MAX_JSON_BYTES) {
    throw new OnlineStoreError(
      "PAYLOAD_TOO_LARGE",
      "요청 내용이 너무 큽니다.",
      413,
    );
  }

  let value: unknown;
  try {
    value = await request.json();
  } catch {
    throw new OnlineStoreError(
      "INVALID_JSON",
      "요청 형식을 확인해 주세요.",
      400,
    );
  }

  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new OnlineStoreError(
      "INVALID_REQUEST",
      "요청 형식을 확인해 주세요.",
      400,
    );
  }
  return value as Record<string, unknown>;
}

export function onlineJson(
  body: Record<string, unknown>,
  init: ResponseInit = {},
): Response {
  const headers = new Headers(init.headers);
  headers.set("cache-control", "no-store");
  return Response.json(body, { ...init, headers });
}

export function onlineApiErrorResponse(error: unknown): Response {
  if (error instanceof OnlineStoreError) {
    return onlineStoreErrorResponse(error);
  }
  if (error instanceof OnlineGameError) {
    return onlineJson(
      {
        error: {
          code: error.code,
          message: error.message,
        },
      },
      { status: gameErrorStatus(error.code) },
    );
  }
  return onlineStoreErrorResponse(error);
}

export function routeRoomCode(context: {
  params: Promise<{ code: string }> | { code: string };
}): Promise<string> {
  return Promise.resolve(context.params).then(({ code }) => code);
}

export function optionalEventSequence(request: Request): number | undefined {
  const raw = new URL(request.url).searchParams.get("sinceEventSeq");
  if (raw === null || raw === "") {
    return undefined;
  }
  const sequence = Number(raw);
  if (!Number.isSafeInteger(sequence) || sequence < 0) {
    throw new OnlineStoreError(
      "INVALID_EVENT_SEQUENCE",
      "이벤트 기준값을 확인해 주세요.",
      400,
    );
  }
  return sequence;
}

function gameErrorStatus(code: string): number {
  if (
    code.includes("NOT_FOUND") ||
    code.includes("UNKNOWN_PLAYER")
  ) {
    return 404;
  }
  if (
    code.includes("INVALID") ||
    code.includes("REQUIRED") ||
    code.includes("MALFORMED")
  ) {
    return 400;
  }
  if (code.includes("FORBIDDEN")) {
    return 403;
  }
  return 409;
}
