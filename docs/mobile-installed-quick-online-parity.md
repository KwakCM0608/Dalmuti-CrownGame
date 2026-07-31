# Installed mobile: quick / online presentation parity

This is the code-level acceptance checklist for the installed phone game.
Network transport is allowed to differ, but the player-visible result must not.
Shared numeric values live in `lib/game-presentation-parity.ts`.

| Area | Quick match reference | Online reference | Required invariant | Status |
| --- | --- | --- | --- | --- |
| Felt / table geometry | `app/globals.css` `.felt-table`, `.play-area` | `app/online/online.module.css` `.table`, `.tableCenter` | Same available felt composition; network header/chat may reserve their own exterior space | Checked |
| Settled table cards | `app/page.tsx` `mobileTableCardStep`; `app/globals.css` `.table-cards` | `app/online/page.tsx` `--table-card-step-small`; terminal installed CSS block | Portrait `88 x 135`, max step `32`, spread `160`, rotation `1.1deg`, lift `0.9`; short landscape `70 x 108` | Shared / fixed |
| Rank seat layout | quick `.opponent-row` installed rail | online `.seatRing[data-mobile-layout="true"]` installed rail | Rank order determines position; viewer occupies their real rank seat; upper/lower row split is identical; both rows sit completely outside the felt | Shared / fixed |
| Rank seat panel | quick `.player-copy` / `.player-count` | online `.playerCopy` / `.handCount` | Same wine-and-bronze panel; nickname remains left aligned and the hand count is always right aligned in both rows | Shared / fixed |
| Local and remote action origins | `PublicTurnActionLayer`: `anchors.players[action.player.id]` | `EventOverlayView`: `stableAnchors.players[fromId || actorId]` | Every card/PASS starts at the acting player's visible rank seat, including the viewer | Checked |
| Public card motion | `publicCardPlay` | `onlinePublicCardPlay` / `onlinePublicCardPlayMobile` | Same `0/10/22/52/72/88/100%` motion stages; overall lock `2250ms`, card motion `2080ms` | Checked |
| Public action-card geometry | quick `.public-play-card` | online terminal installed `.playOverlay` rule | `92 x 142`; expanded max step `54 / 190`; settled max step `24 / 190`; origin spread `9`; delay max `36 / 100` | Shared / fixed |
| PASS motion | `publicPassToTable` | `onlinePublicPassToTable` | Same `0/12/24/58/100%` path; overall lock `1500ms`, badge motion `1380ms` | Checked |
| Previous pile during motion | quick `visibleTable` from `publicAction.previousTable` | online `visibleTable` from `activeEvent.data.previousTable` | New cards never appear settled before motion finishes; the last PASS reaches center before clearing | Checked |
| Hand reveal intro | quick reveal phases | online `MATCH_STARTED` / hand reveal phase | Every assigned slot is present face-down; initial deal is `520ms` with `30ms` stagger; intro `2400ms`; hand reveal phase `1400ms`; per-card flip path is identical | Shared / fixed |
| Rank draw / confirmation | quick rank phases | online rank phases / engine durations | Intro `3300ms`; reveal pause `1500ms`; reveal `3400ms`; confirmation `2600ms` | Pinned |
| Tax | quick tax intro / `TaxTransferLayer` | online tax events / `taxCardTransfer` | Intro `2400ms`; each stage `6000ms`; visible card transfer `5550ms`; private identities remain private | Pinned |
| Revolution | quick revolution overlays | online revolution events / overlays | Announcement `3300ms`; great-revolution swap `2600ms`; persistent red field is state-driven, not overlay-lifetime-driven | Pinned |
| DALMUTI | quick DALMUTI action layer | online `DALMUTI_EFFECT` overlay | Overall `3300ms`; field `3250ms`; card `3100ms`; automatic PASS `2550ms`; banner `3050ms`; mobile offset `34`, first delay `360ms`, stagger `90ms` | Shared / fixed |
| Rank movement / result handoff | quick rank transition | online rank move runtime | Movement `2300ms`; result waits for movement completion; no seat teleport | Pinned |
| Turn timer | quick turn timer | online turn deadline timer | `30000ms`; urgent state begins at `10000ms`; action controls stay locked during public presentation | Pinned |
| Own dock / action panel | quick `.human-zone`, `.turn-controls` | online `.ownDock`, `.actionBar` | Phone hand card `53 x 81`, overlap `-24`, dock min height `108`, action panel min height `44` | Checked |
| Finished-hand state | quick `.finished-hand-state` | online `.finishedHand` | Same medal/copy hierarchy and no unrelated temporary hand | Checked |
| Chat / emotes | none | online `OnlineChatPanel` | Online-only exception; may float/drag/collapse but must not change felt anchors or action timings | Intentional exception |
| Room / reconnect UI | none | online lobby and connection state | Online-only exception; excluded from visual parity | Intentional exception |

## Regression rule

`tests/mobile-installed-parity.test.mjs` must fail when a shared timing,
installed-card geometry, motion origin, public-motion stage, or phone dock
dimension drifts in only one mode.
