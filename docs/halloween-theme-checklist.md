# Halloween theme implementation checklist

This document is the acceptance checklist for the optional Halloween visual
theme. The theme is a local presentation preference. It must never change game
rules, server snapshots, hidden information, timers, animation ordering, or bot
decisions.

## Theme architecture and reversibility

- [ ] `original` remains the default when no preference has been saved.
- [ ] `halloween` and BGM preferences are stored only in the versioned local
      preferences key.
- [ ] Invalid or partially written local storage data falls back to safe
      defaults.
- [ ] The initial document theme is applied before hydration so an original
      palette does not flash before Halloween appears.
- [ ] The three decorative cards on the online entry screen are painted from
      the pre-hydration document theme; Halloween never briefly exposes the
      Original card faces during a direct visit, reload, or slow hydration.
- [ ] Changing the theme updates the current screen immediately without a
      reload.
- [ ] Changing back to Original restores every original card, card back, crown,
      table color, surface color, border, text color, and effect color.
- [ ] A preference change in another tab is reflected through the `storage`
      event.
- [ ] Theme state is not included in quick-match game state, online commands,
      D1 room state, or player snapshots.
- [ ] Theme changes do not remount or reset an active game.

## Settings screen

- [ ] On desktop, a keyboard-focusable gear icon appears in the upper-right of
      the main four-option panel without changing the panel grid or click
      targets.
- [ ] On mobile, the desktop panel gear is hidden and the existing top-bar gear
      remains available without overlapping the brand or safe area.
- [ ] The gear has the accessible name `환경설정` and opens `/settings`.
- [ ] The settings screen has clear navigation back to the main screen.
- [ ] BGM has an on/off control.
- [ ] BGM volume accepts values from 0 through 100 and displays the current
      value.
- [ ] BGM controls persist, but no audio is downloaded or played until a BGM
      source is added in a later update.
- [ ] Original and Halloween are exposed as one accessible radio group.
- [ ] Both theme choices show an accurate card preview and selected state.
- [ ] The settings page itself changes palette and crown with the selected
      theme.

## Halloween assets

- [ ] `public/cards/halloween/01.webp` through `12.webp` exist at 1040 x 1600.
- [ ] Rank 13 uses `public/cards/halloween/joker.webp` at 1040 x 1600.
- [ ] `public/cards/halloween/back.webp` exists at 1040 x 1600.
- [ ] The back artwork fills its complete 1040 x 1600 canvas; it is not a small
      thumbnail anchored in the upper-left of an otherwise empty image.
- [ ] The Halloween back keeps all four corner marks and the central mark
      visible at desktop and mobile card sizes.
- [ ] The Halloween back uses a low-contrast charcoal ink-wash texture rather
      than a flat digital-black fill, without reducing symbol contrast.
- [ ] `public/themes/halloween/crown.webp` is a square crop derived from the
      Halloween Dalmuti crown.
- [ ] `public/themes/halloween/dalmuti-hand-field-atlas-v2.png` is a 4 x 3 atlas
      of the twelve exact hands cut from the approved reference. It preserves
      their scale and facing direction while spreading the crowd across a
      1200px-wide field. The two largest mirrored hands are promoted above the
      crowd, inset as a pair, and vertically normalized so their fingertips
      start at the same height while their palms cradle the card's lower-left
      and lower-right edges like the hands holding the crown in card 1. No
      extra foreground hand may cross or obscure the card artwork.
- [ ] `public/themes/halloween/ink-wash-field-texture-v2.webp` preserves the
      Halloween card back's charcoal paper grain at 1536 x 1024 without its
      symbols or border, so the Revolution field stays crisp at desktop size.
- [ ] Card rank, action legality, and card identity continue to use numeric game
      data rather than artwork filenames.

## Shared card presentation

- [ ] Quick-match hand faces use the selected artwork.
- [ ] Quick-match opening rank faces use the selected artwork.
- [ ] Quick-match opening rank backs use the selected back.
- [ ] Quick-match concealed hands and reveal animations use the selected back.
- [ ] Quick-match opponent hand-count mini cards use the selected back.
- [ ] Quick-match tax-transfer private cards use the selected back.
- [ ] Online hand faces use the selected artwork.
- [ ] Online opening rank faces and backs use the selected artwork.
- [ ] Online concealed hands, seat reveal cards, and reveal animations use the
      selected back.
- [ ] Online remote players' mini card backs use the selected back.
- [ ] The rulebook's face cards and rank-draw back use the selected artwork.
- [ ] Rank 1 and Joker face-border styling uses `data-rank="1"` and
      `data-rank="13"`, never an artwork URL, so it works in both theme
      directories.
- [ ] Online `.card:not(.cardBack)[data-rank="1"]` and rank 13 selectors keep
      `.cardBack` excluded so a hidden rank 1 or Joker never renders as a black
      face.
- [ ] Every Halloween face-card wrapper uses the same black frame, including
      ranks 2 through 12, opening-rank cards, confirmation cards, tax/action
      cards, and rulebook examples; Original retains its existing frames.
- [ ] Every Halloween card-back surface uses the full-frame shared back image
      with centered cover sizing in Quick, Online, and the rulebook.
- [ ] Every visible card profession name follows the selected artwork: Original
      keeps its existing Korean names and Halloween uses the translated names
      printed by the Halloween deck, without renaming player social ranks.

## Main and quick-match palette

- [ ] The page background, top bar, dividers, focus rings, and buttons use the
      Halloween neutral-charcoal palette.
- [ ] The main hero, menu cards, quick setup panel, selectors, descriptions,
      and primary action use the Halloween palette.
- [ ] The main crown and upper-left brand seal use the Halloween crown.
- [ ] The central felt field is gray in Halloween and green in Original.
- [ ] Score and history rails, section headings, rank rows, chip labels, and
      scrollbars use the selected palette.
- [ ] All player seats, current-player emphasis, status badges, timers, and
      hand-count labels retain sufficient contrast.
- [ ] Pass, submit, skip, sidebar, and rulebook controls retain hover, pressed,
      disabled, and focus-visible states.
- [ ] Opening-rank, hand-reveal, tax, revolution, great-revolution, Dalmuti,
      round-movement, and result overlays follow the explicit theme contracts
      below without changing the game-action lock or event order.
- [ ] Neutral pregame states (opening-rank, hand reveal, no-tax, tax captions,
      and game start) use charcoal/violet rather than the Original green/gold
      palette.
- [ ] Both revolution types begin on the neutral gray field: seven staggered
      tapered, highlighted ink drops fall with a slight wobble and squash on
      impact. On the exact contact frame, the same photographed-style alpha
      mask appears as the small impact mark and immediately continues soaking
      outward through a cumulative canvas; no separate ellipse, ripple, or
      mismatched landing graphic intervenes. Every bloom starts at its exact
      drop coordinate, retains the dense pool, absorbed scalloped edge,
      capillary branches, and detached micro-specks, and receives independent
      rotation and stretch. The canvas remains mounted after all
      blooms overlap, so there is no polygonal wipe, layer-removal flash,
      whole-field fade, or Original red transition layer.
- [ ] After the shared one-shot contamination transition, a great revolution
      uses the Original-style full-field clockwork effect for the rest of the
      act. The clock face and tick marks stay fixed while only its hands run
      counter-clockwise; no bounded clock face or persistent bubbling remains.
- [ ] A Halloween Dalmuti submission removes the gold sparkle treatment and
      reveals all twelve hands cut from the approved reference one at a time
      across nearly the full lower field. The largest hand visibly supports the
      card while its fingers overlap only the lower card edge.
- [ ] The card reveal and automatic-PASS gathering start together; PASS keeps
      the Original seat-to-center path and arrangement while only its timing
      and Halloween colors change.
- [ ] Revolution and danger feedback remain visually distinct from the normal
      Halloween palette.

## Online-only palette

- [ ] Online entry, create-room, and join-room surfaces use the selected theme.
- [ ] The online header brand seal and entry crown use the Halloween crown.
- [ ] Lobby player slots, ready states, bot difficulty picker, invite code, and
      host controls use the selected theme.
- [ ] Network status, reconnecting, error, and pending states remain legible.
- [ ] The online game field is gray in Halloween and green in Original.
- [ ] Online seats, action log, turn timer, pass/submit controls, and result
      dialog use the selected palette.
- [ ] Chat panel, input, message history, emote picker, and nickname emotes use
      the selected palette without moving the table center or animation origin.
- [ ] Online opening-rank, hand reveal, no-tax, tax, and game-start presentation
      surfaces use the same neutral Halloween palette as Quick Match.
- [ ] Online normal revolution uses the black-ink field treatment, while great
      revolution keeps the ink transition and the same fixed-face,
      counter-clockwise-hand clockwork contract as Quick Match.
- [ ] Online Halloween Dalmuti uses the same sequential hand -> largest hand ->
      card plus PASS contract as Quick Match without changing the remote-action
      presentation queue.
- [ ] Different clients may use different local themes in the same room without
      affecting the shared revision or server state.

## Dialogs and secondary surfaces

- [ ] Rulebook backdrop, dialog, navigation, examples, buttons, cards, and crown
      are themed.
- [ ] Credits backdrop, dialog, labels, links, button, and crown are themed.
- [ ] PWA install/update prompts retain readable contrast in both themes.
- [ ] The safe-area background and browser theme color follow the selected
      theme while the page is open.
- [ ] The offline fallback reads the same local theme preference, uses the
      Halloween crown and charcoal/purple/orange palette when selected, and
      keeps the existing reconnect behavior unchanged.

## Responsive and installed-app verification

- [ ] Desktop quick match is checked at 1440 x 900 in both themes.
- [ ] Desktop online entry, lobby, and game are checked at 1440 x 900.
- [ ] Mobile web is checked at 390 x 844.
- [ ] Installed Android standalone/fullscreen presentation is checked without
      changing its existing layout or transition behavior.
- [ ] iPhone standalone presentation is checked with safe-area insets.
- [ ] Theme controls do not introduce horizontal scrolling, clipped seats, or
      changes to card movement origins.

## Cache, tests, and release gate

- [ ] The service-worker cache version is bumped for the theme release.
- [ ] The Halloween back, crown, and Dalmuti hand sprite sheet are precached;
      face cards continue to use the same runtime cache policy as Original face
      cards.
- [ ] Online API requests and all mutation requests continue to bypass caches.
- [ ] Preference parsing, clamping, artwork paths, and theme colors have unit
      coverage.
- [ ] Static wiring tests cover main, settings, quick match, online, rulebook,
      dialogs, assets, and service-worker paths.
- [ ] All existing quick-match and online engine regression tests pass.
- [ ] Theme is switched Original -> Halloween -> Original during manual QA,
      including after a reload and during an active game.
- [ ] Manual QA visits every Quick and Online card-back location, verifies black
      Halloween face frames, reloads the online entry screen to check for card
      flash, and exercises normal revolution, great revolution, and Dalmuti in
      both themes.
- [ ] `pnpm test`, `pnpm run lint`, `pnpm typecheck`, and `git diff --check`
      pass before release.
