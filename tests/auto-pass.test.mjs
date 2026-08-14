import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const { hasLegalCardPlay } = await import(
  new URL("../lib/auto-pass.ts", import.meta.url)
);

test("auto PASS only triggers when no legal response exists", () => {
  const table = { rank: 8, count: 2 };

  assert.equal(hasLegalCardPlay([{ rank: 7 }, { rank: 13 }], table), true);
  assert.equal(
    hasLegalCardPlay([{ rank: 9 }, { rank: 9 }, { rank: 13 }], table),
    false,
  );
  assert.equal(hasLegalCardPlay([{ rank: 13 }, { rank: 13 }], table), false);
});

test("a non-empty hand can always lead a cleared table", () => {
  assert.equal(hasLegalCardPlay([{ rank: 12 }], null), true);
  assert.equal(hasLegalCardPlay([{ rank: 13 }, { rank: 13 }], null), true);
  assert.equal(hasLegalCardPlay([], null), false);
});

test("collapsed online chat surfaces unread incoming messages", async () => {
  const [page, styles] = await Promise.all([
    readFile(new URL("../app/online/page.tsx", import.meta.url), "utf8"),
    readFile(
      new URL("../app/online/online.module.css", import.meta.url),
      "utf8",
    ),
  ]);
  assert.match(page, /lastReadMessageId/);
  assert.match(page, /messages\s*\.slice\(boundaryIndex \+ 1\)/);
  assert.match(page, /message\.playerId !== viewerId/);
  assert.match(page, /styles\.chatUnreadBadge/);
  assert.match(page, /새 메시지 \$\{unreadCount\}개/);
  assert.match(styles, /\.chatPanelUnread\.chatPanelCollapsed/);
  assert.match(styles, /@keyframes chatUnreadPulse/);
});

test("mobile install prompt promotes stable app play", async () => {
  const lifecycle = await readFile(
    new URL("../app/components/PwaLifecycle.tsx", import.meta.url),
    "utf8",
  );
  assert.match(lifecycle, /더 안정적으로 플레이할 수 있어요/);
  assert.match(lifecycle, /화면 잘림 없이/);
});
