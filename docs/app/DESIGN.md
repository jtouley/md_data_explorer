# Renderer design direction

Status: adopted for the Electron renderer (`electron/index.html`, `electron/src/index.css`,
`electron/src/renderer.js` and the panel modules). This page is the contract the CSS and the
renderer implement. Values here are the source of truth; change them here first.

## The brief

The owner asked for an app that feels like a conversation-first assistant, but for clinicians
rather than programmers, with a dark mode, and with **evidence-backed reasoning, always**.

We borrow interaction and layout *patterns* that conversation-first apps share: one centred
reading column, a composer docked at the bottom, a left sidebar with past conversations titled
by their first question, quiet chrome. We copy no product's name, logo, brand colours, icons or
assets. The sage accent, the warm neutrals and the type pairing are this app's own.

Four independent audits, each applying one designer's published skill set, were rendered in
Playwright against the pre-redesign renderer (`m2int`). Their reports and screenshots live in
the session scratchpad under `design_audit/{mengto,owl,conardli,kowalski}`. The skills are
criteria, not authority: where they disagree, the decision below rests on measured evidence or
on what a clinician has to do with the screen.

## Where all four audits agree

| Finding | Evidence (audit) | Decision |
|---|---|---|
| No dark mode at all | Light and dark renders byte-identical (all four) | Full semantic token set, light + dark, System/Light/Dark toggle |
| Chat squeezed by panels | 48 px of chat at 1280×820 under a 454 px preview and a 215 px enrichment block (ConardLi, Kowalski); ~115 px (MengTo); ~40 px (Owl) | Dataset preview, metadata suggestions and change history move to a dataset panel that opens from the sidebar. The chat owns the main column |
| Message states look alike | Answer, hold, refusal and crash rendered with the same box (Owl, ConardLi, Kowalski) | Each state has its own shape, a word and an icon, not colour alone |
| Plan line is the smallest text on the card | 12.8 px mono (Owl, ConardLi) | Plan shown as a 16 px labelled list in the body face |
| Mono, left-aligned table; literal "suppressed" reads as data | All four | Body face with tabular figures, numbers right-aligned, hidden cells muted with a footnote |
| Emoji and glyph icons, generic copy | 📊, ◉, ⟳, →, ✕, ✓/✗; "Claim:" prefix; "Group comparison" twice (all four) | One small inline stroke-icon set; claim is the headline; the intent label shown once as a caption |
| Centred column, docked composer | 720–760 px (all four) | 760 px column, composer docked and pinned |
| Sessions titled by first question | Every session read "Dataset: ds_cohort" (Owl, ConardLi, MengTo) | See "Session titles" below for what is possible today |
| Claim as headline + quiet stats line | MengTo, Kowalski, ConardLi | Headline sentence, then one stats line, then caveats; split done in the renderer without changing a word or number |
| Held plan as a draft awaiting approval | All four | Tinted draft card with its own heading and two plain-language buttons |
| Refusal is a calm notice, not an error | All four | Neutral notice with "Not answered" label; errors keep the danger colour |
| Bugs | `.thinking-dots span` also matches `.thinking-stage` (one letter per line, Kowalski frame `stream-11-1302ms.png`); `white-space: pre-wrap` blank bands; Arial buttons; doubled quotes on examples; rogue `#2563eb` hover; composer disabled while streaming; new answer below the fold | All fixed (see the re-audit table in the session report) |

## Where they disagree, and what we chose

### Held card: amber left rule, or tinted card with a border?

MengTo and Kowalski proposed an amber/accent left rule; ConardLi proposed a tinted card with a
1 px border and no stripe. **Decision: tinted card + 1 px border, no stripe.**
Evidence: MengTo's own `no-ai-design-slop` checklist lists "a thick accent stripe is attached to
one side of a rounded card" as a slop pattern (`ARTICLE.md`, line 88), so the stripe contradicts
the skill that proposed it. Functionally, a left rule is also the shape of a quotation or a
callout; a *draft* is better signalled by a different surface plus a heading that says what to
do ("Check this plan before it runs"). The border passes 3:1 against its tint in both themes
(3.51 light, 3.10 dark).

### Fonts: Source Serif 4 + Source Sans 3, or IBM Plex?

MengTo ranked Source Serif 4 + Source Sans 3 first; ConardLi proposed IBM Plex Sans + Plex Mono.
**Decision: Source Sans 3 for all interface text and tables; Source Serif 4 only for the claim
headline and the empty-state greeting. No mono face.**
Evidence: the audits agree the mono plan line and mono table are the problem, so a mono face has
no remaining job. Source Serif was drawn as a companion to Source Sans, so the pair shares
proportions and needs no tuning. Source Sans 3 ships tabular lining figures (`tnum`), which the
right-aligned tables rely on. A serif headline gives the one sentence that matters a distinct
voice from the interface around it, which is the "one authored moment" MengTo asks for. Both
are SIL OFL 1.1 and were taken from Adobe's own repositories (see "Fonts and licences").

### Dark base: #121413, #161513 or #141412?

Owl proposed a green-tinted `#121413`, ConardLi a warm `#161513`, Kowalski a neutral-warm
`#141412`, MengTo `#1B1A18`. **Decision: `#161513` (warm).** The light theme is a warm paper
(`#F7F6F3`); a warm dark keeps the two themes the same app. A green-tinted base lowers the
separation of the sage accent, which is the only hue that marks "act here". Elevation is shown by
lightness steps (`#161513` → `#1B1A18` → `#211F1D` → `#2C2A27`), not shadows (Owl,
ConardLi, Kowalski agree on that).

### Refused: blue "info", or neutral?

Owl proposed blue (`#33598a` / `#9db8e0`); ConardLi a slate neutral; Kowalski "calm, not red".
**Decision: neutral.** Uses the secondary text colour on the sidebar surface, with an info
icon and the words "Not answered". A refusal on privacy grounds is the product working, so it
must not borrow the error's red; a second accent hue (blue) would compete with sage, which
already means "you can act". The word and icon carry the meaning (WCAG 1.4.1).

### User question: muted line or bubble?

MengTo proposed the question as a small muted line above each answer; ConardLi a right-aligned
bubble. **Decision: right-aligned bubble on the sunken surface, max 80 % width.** The chain a
clinician audits is question → plan → answer. A muted line makes their own words look like
metadata; the bubble keeps turn-taking legible when scanning back.

### Status pills on every message?

Owl proposed "Answered ✓ / Needs your OK / Not answered / Error" pills on every message.
**Decision: label only the states that need different action** (held, refused, error, stopped).
An answer is shown by its claim; a pill saying "Answered" on every turn is noise (MengTo's
restraint criterion). Answers get the analysis type ("Group comparison") as a caption.

### Hidden cells: "—", "hidden", or "<11"?

MengTo proposed "—" with a footnote "fewer than 11 patients"; ConardLi "<11" with †; Owl a muted
italic "hidden". **Decision: muted italic "hidden" with a † and one footnote; cells that the API
already sends as "<11" keep that text, muted, with the same †.**
Evidence: the API marks *both* primary small cells (1–10) *and* secondary cells (hidden so a small
cell cannot be worked out from a total) with the same word, deliberately, so the two cannot be
told apart (`result_presentation.py`, `_mask`). Rendering every hidden cell as "fewer than 11"
would be false for secondary cells; rendering "—" reads as "missing" or "zero" in clinical
tables. The footnote says what is true for both: "Hidden to protect privacy: counts from 1 to 10
are not shown, and some other cells are hidden so those counts can't be worked out from totals."
We never write "<5"; the threshold is 11.

**Data check (cohort details) uses one label only.** Its report (`contracts/quality.py`) shows
every suppressed count the same way, as "some … ; count hidden to protect privacy", never "<11":
with a visible row total, two labels would tell a reader that the non-"<11" hidden counts are 11
or more, and a column's missing + bad + valid counts would then pin a small one. A hidden count
there only means "at least 1" (zero is shown as 0). Its footnote: "Hidden to protect privacy:
counts from 1 to 10 are not shown, and some larger counts are hidden too so those can't be
worked out from the totals. A hidden count is never zero."

### Session titles

All audits ask for sessions titled by their first question. The API does not persist messages
(`message_count` is always 0 for real sessions; conversation history is in memory), so there is
no first question to show after a restart. **Decision:** sessions created in this run are titled by
their first question (held in memory only); others read "Earlier conversation". The renderer
also reads a `title` field if the API ever sends one. We rejected saving question text in
`localStorage`: questions can identify patients, and a copy there would survive "Delete
dataset", which promises to remove every session for that dataset
(`docs/security/LOCAL_DATA_SECURITY.md`). Follow-up for the API: persist the first question with
the session so deleting the dataset deletes it too.

## Tokens

All colours are semantic custom properties on `:root`, overridden under
`:root[data-theme="dark"]` and `@media (prefers-color-scheme: dark) :root:not([data-theme="light"])`.
`color-scheme: light dark` lets native controls and scrollbars follow.

| Token | Light | Dark | Use |
|---|---|---|---|
| `--bg` | `#F7F6F3` | `#161513` | Canvas, chat column |
| `--surface-1` | `#EFEDE8` | `#1B1A18` | Sidebar, refusal notice |
| `--surface-2` | `#FFFFFF` | `#211F1D` | Composer, tables, panels |
| `--surface-3` | `#FFFFFF` + shadow | `#2C2A27` | Menus, sheets |
| `--sunken` | `#ECEAE4` | `#272522` | User question bubble, table head |
| `--border` | `#E3E0D9` | `#33302C` | Decorative hairlines (not a UI boundary) |
| `--border-strong` | `#847F75` | `#77726A` | Control boundaries (inputs, select, secondary buttons) |
| `--text-1` | `#1C1B19` | `#ECEAE5` | Body, headline |
| `--text-2` | `#4A4844` | `#BDB9B1` | Secondary text, stats line |
| `--text-3` | `#68655F` | `#9A968E` | Captions, meta, hidden cells |
| `--accent` | `#3F6E4E` | `#7FB08E` | Primary action, focus, active session |
| `--accent-hover` | `#345C41` | `#96C2A3` | Primary hover |
| `--accent-subtle` | `#E4ECE5` | `#22302A` | Selected rows, chips hover |
| `--on-accent` | `#FFFFFF` | `#10140F` | Text on primary |
| `--held` / `--held-bg` / `--held-border` | `#8A5A12` / `#FBF3E4` / `#A57A33` | `#E0B46A` / `#2A2318` / `#8A6A32` | Plan awaiting approval |
| `--notice` / `--notice-bg` | `#4A4844` / `#EFEDE8` | `#BDB9B1` / `#1B1A18` | Refusal |
| `--danger` / `--danger-bg` | `#A8373A` / `#FBEDEC` | `#E58C86` / `#2E1C1B` | Errors, delete |

### Measured contrast (WCAG 2.x relative luminance; re-computed from `index.css` by `electron/src/designTokens.test.js`)

Minimums: 4.5:1 for text, 3:1 for UI boundaries. All pairs below pass.

| Pair | Light | Dark |
|---|---|---|
| text-1 on bg / surface-1 / surface-2 / sunken | 15.92 / 14.71 / 17.21 / 14.31 | 15.18 / 14.46 / 13.66 / 12.71 |
| text-2 on bg / surface-1 / surface-2 / sunken | 8.44 / 7.80 / 9.12 / 7.58 | 9.33 / 8.89 / 8.40 / 7.82 |
| text-3 on bg / surface-1 / surface-2 / sunken | 5.37 / 4.96 / 5.81 / 4.83 | 6.19 / 5.90 / 5.58 / 5.19 |
| accent (as text) on bg / surface-1 / surface-2 | 5.47 / 5.05 / 5.91 | 7.39 / 7.04 / 6.65 |
| accent on accent-subtle | 4.91 | 5.58 |
| on-accent on accent / accent-hover | 5.91 / 7.63 | 7.53 / 9.34 |
| held on held-bg; text-1 on held-bg | 5.36; 15.61 | 8.07; 12.92 |
| notice on notice-bg | 7.80 | 8.89 |
| danger on bg / surface-2 / danger-bg | 5.94 / 6.42 / 5.63 | 7.32 / 6.59 / 6.49 |
| border-strong on bg / surface-1 / surface-2 (UI, ≥3) | 3.68 / 3.40 / 3.98 | 3.82 / 3.64 / 3.44 |
| held-border on held-bg (UI, ≥3) | 3.51 | 3.10 |

`--border` is decorative only (hairlines between rows); every interactive boundary uses
`--border-strong` or the accent.

### Type

| Token | Size / line-height | Face, weight | Use |
|---|---|---|---|
| `--fs-xs` | 12 / 16 | Sans 500 | Captions, date group headings, footnotes |
| `--fs-sm` | 13 / 18 | Sans 400 | Stats line, meta, table head |
| `--fs-md` | 14 / 20 | Sans 400–600 | Sidebar, buttons, table cells |
| `--fs-body` | 16 / 25 | Sans 400 | Question, notices, plan, composer |
| `--fs-claim` | 20 / 28 | Serif 500 | Claim headline |
| `--fs-display` | 28 / 34 | Serif 500 | Empty-state greeting |

Weights 400/500/600 only. Sentence case everywhere (no tracked uppercase). Numbers in tables
use `font-variant-numeric: tabular-nums lining-nums`.

### Space, radius, elevation

- Spacing: 4, 8, 12, 16, 24, 32, 48, 64 (`--sp-1` … `--sp-8`). Nothing off this ladder.
- Radii: 6 controls, 10 cards/tables/panels, 16 composer, full for chips.
- Elevation: light uses `--shadow-1` on the composer and `--shadow-2` on menus/sheets; dark uses
  no shadows, only a lighter surface plus a 1 px border.
- Layout: sidebar 264 px; chat column max 760 px with 24 px side padding; dataset panel 440 px
  sheet from the right. Under 1000 px wide the sidebar narrows to 232 px; the column stays
  fluid.

### Motion (after Kowalski)

| Token | Value |
|---|---|
| `--ease-out` | `cubic-bezier(.23, 1, .32, 1)` |
| `--ease-in-out` | `cubic-bezier(.77, 0, .175, 1)` |
| `--dur-press` | 100 ms |
| `--dur-hover` | 120 ms |
| `--dur-pop` | 180 ms |
| `--dur-enter` | 220 ms |
| `--dur-exit` | 140 ms |
| `--press-scale` | .97 |

Choreography:
- **Send**: button presses to `scale(.97)` (transform only). The question bubble enters at once
  (opacity + 4 px rise, 180 ms).
- **Working**: three dots pulse opacity on a 1.2 s cycle; the stage text ("Reading your
  question", "Running the analysis") is a sibling, static, and swaps text without moving. After
  3 s an elapsed counter appears.
- **Answer lands**: the card enters with opacity 0→1, `translateY(6px)`→0, `blur(2px)`→0 over
  220 ms; caption, claim, stats and table stagger by 40–60 ms. Numbers never count up.
- **Held card**: same entry. After "Run this analysis" the plan stays and the result arrives
  below; after "Not what I meant" the buttons are replaced by a one-line note.
- **Scroll**: when the reader was at the bottom, a new answer is scrolled into view; if the
  answer is taller than the view, its top is aligned instead so the headline is never off-screen.
- **Hover** effects are gated behind `(hover: hover)`. Transitions name their properties (no
  `transition: all`).
- **Reduced motion**: no translate, no blur, no pulse; opacity fades of 150 ms remain, scroll
  is instant.

## Message states

| State | Shape | Label / icon | Copy |
|---|---|---|---|
| Answered | No box; caption, serif headline, stats line, caveats, table, footnote | Analysis type as caption | Claim text from the API, unchanged |
| Held | Tinted draft card with border | Clock icon + "Check this plan before it runs" | API reason, then the plan as a labelled list, the exact plan text on demand, buttons "Run this analysis" / "Not what I meant" |
| Confirmed | Held card, buttons replaced | "You approved this plan." | Result streams below |
| Rejected | Held card, buttons replaced | "Not run." | "Not run. Rephrase your question below." |
| Refused | Neutral notice | Info icon + "Not answered" | API reason, then "Your question is back in the box below so you can change it." |
| Error | Danger-tinted notice | Alert icon + "Something went wrong" | API message or a plain fallback; question restored |
| Loading | Dots + stage text | — | "Reading your question", "Running the analysis", "· 4 s" after 3 s |
| Streaming | Send becomes Stop; composer stays editable | Stop icon | Enter while busy: "Wait for this answer, or press Stop." |
| Stopped | Muted note | "Stopped" | "Stopped waiting for this answer." |
| Empty (no cohort) | Centred | — | "Choose a cohort to begin." + Upload / Use the sample cohort |
| Empty (cohort chosen) | Centred serif greeting + chips | — | "What would you like to know about {cohort}?" + three questions built from its columns (fill the box, never auto-send) |
| Offline | Banner at the top of the main column; empty state explains | Alert icon | "Can't reach the analysis engine on this computer." + Try again; empty state "Waiting for the analysis engine." |
| Deleting | In-page dialog under the cohort switcher | — | Existing delete copy; "Deleting…" while busy |

## Theme

- Three choices: System, Light, Dark, as a segmented radio group in the sidebar footer.
- Electron: the renderer calls `window.appTheme.set(source)` (preload → `ipcMain.handle`), main
  sets `nativeTheme.themeSource`, which drives `prefers-color-scheme` in the page, and persists
  the choice to `settings.json` in the app's user-data folder. Main reads it before the window is
  created and sets the window `backgroundColor` to match, so the first frame is already the right
  theme. Only the app's own page may call it, and only these three values are accepted.
- Browser (Vite, Playwright): an inline script in `<head>` reads `localStorage` (wrapped in
  `try`) and sets `data-theme` before first paint; the stylesheet does the rest.
- Theme change cross-fades background and text colours for 200 ms (none under reduced motion).

## Navigation

- **Sidebar** (top to bottom): app name; cohort switcher (native `<select>`, so keyboard and
  screen readers work as before) with a ⋯ button opening a small menu: Upload a file, Use the
  sample cohort, Preview and details, Refresh the list, Delete this cohort; a meta line
  "400 rows · 14 variables" (rows, not "patients": a file can hold several rows per patient,
  and the meta line must not claim what it has not checked); a "New question" button; past conversations grouped by
  Today / Yesterday / Previous 7 days / Older; footer with the on-this-computer status and the
  theme switch.
- **Main**: the chat column and the docked composer only. An offline banner appears only when
  the engine cannot be reached; the "Backend connected" badge is gone.
- **Dataset panel**: a sheet from the right with the preview table, column suggestions and
  change history. Opened from the ⋯ menu or the meta line; Escape or the close button returns
  focus to where it came from.

## Copy deck

Plain clinical language; "cohort" for the uploaded dataset, "rows" for rows (the API's own claims
say "patients" where it has counted patients), no internal ids on screen.

| Where | Before | After |
|---|---|---|
| Title | Clinical Analytics + ◉ | Clinical Analytics (text only) |
| Status (connected) | Backend connected | Running on this computer |
| Status (offline) | Backend unavailable | Can't reach the analysis engine on this computer. |
| LLM banner, not reachable | Local LLM (Ollama) is not reachable at … Tier-3 NL fallback … | Some questions need the local language model, which isn't running. Simple questions still work. |
| LLM banner, model missing | Ollama is running but the default model "…" was not found … | The local language model "{model}" isn't installed, so some questions won't work. Simple questions still work. |
| Sidebar heading | SESSIONS / + New | (date groups) / New question |
| Session row | Dataset: ds_cohort · 6 msgs · 5m ago | {first question} or "Earlier conversation" / {cohort} · 5 min ago (the message count was dropped: the API always reports 0) |
| Session delete | ✕ + window.confirm | Icon button "Delete conversation: {title}", in-row confirm "Delete this conversation? It can't be recovered." Keep / Delete (focus on Keep; Escape keeps) |
| Dataset select placeholder | Select a dataset... | Choose a cohort… |
| Dataset meta | 400 rows • 14 columns | 400 rows · 14 variables (sidebar; opens the details panel) |
| Upload | Upload dataset / Load sample / ✓ x uploaded / ✗ Upload failed: | Upload a file… / Use the sample cohort / "{name} is ready." / "Couldn't upload {file}: {reason}" |
| Delete | Delete dataset | Delete this cohort… (menu) |
| Welcome | 📊 Welcome to Clinical Analytics … "Try asking:" | No cohort: "Choose a cohort to begin." With cohort: "What would you like to know about {cohort}?" |
| Placeholder | Ask a question about your data... | Ask about {cohort}… |
| Hint | Press Enter to ask; Shift+Enter for a new line | Enter to ask · Shift+Enter for a new line |
| Hint (none) | Select a dataset to start asking questions | Choose a cohort to begin. |
| Working | Processing… / Stage: planning | Reading your question… / Running the analysis… |
| Claim | Claim: {claim} + title twice | {headline} / {stats line} / {caveats}; title once as caption |
| Hidden cell | suppressed | hidden† / <11† + footnote |
| Upload ready + data check | ✓ x uploaded | "{name} is ready." + "Data check found N things to review in cohort details." / "Data check found nothing to flag." / "Data check could not run." |
| Cohort details: data check | (none) | Section "Data check": the API's lines as a list (suppressed counts read "some …; count hidden to protect privacy"), the data-check footnote; on failure "Data check could not run: {reason}. You can still ask questions." |
| Hold heading | (none) | Check this plan before it runs |
| Confirm / Reject | Confirm / Reject | Run this analysis / Not what I meant |
| Rejected | Query rejected — not executed. | Not run. Rephrase your question below. |
| Confirm while busy | Wait for the current answer to finish, then confirm. | Wait for the current answer to finish, then run this. |
| Refusal | Error: … / The question could not be answered: … | Not answered + reason + "Your question is back in the box below so you can change it." |
| Crash | Sorry, I encountered an error: … | Something went wrong + message |
| Lost stream | Connection lost. Please try again. | The connection to the analysis engine dropped before the answer arrived. Ask again. |
| Session switched | Session restored. Switched to session sess_… | Continuing an earlier conversation. Earlier answers aren't shown here; new questions continue it. |

## Fonts and licences

Bundled in `electron/src/fonts/`, unmodified, served from the app (no network at runtime).

| File | Source (primary) | Version | Licence |
|---|---|---|---|
| `SourceSans3VF-Upright.woff2` | `adobe-fonts/source-sans`, branch `release`, `WOFF2/VF/SourceSans3VF-Upright.ttf.woff2` | 3.3408 | SIL OFL 1.1 |
| `SourceSans3VF-Italic.woff2` | same repository, `WOFF2/VF/SourceSans3VF-Italic.ttf.woff2` | 3.3408 | SIL OFL 1.1 |
| `SourceSerif4VF-Roman.woff2` | `adobe-fonts/source-serif`, branch `release`, `WOFF2/VAR/SourceSerif4Variable-Roman.ttf.woff2` | 4.328 | SIL OFL 1.1 |

The licence files were fetched from the same repositories (`LICENSE.md`) and are committed as
`electron/src/fonts/SourceSans3-OFL.md` and `SourceSerif4-OFL.md`. Both name the Reserved Font
Name "Source": the files are shipped unmodified (no subsetting or renaming), which the OFL permits
when bundled with software, together with the licence.

## Known limits

- Session titles after a restart need the API to persist the first question (see above).
- Earlier answers in a resumed session are not shown: the API has no transcript endpoint.
- The confirm request is built in one function (`buildConfirmRequest` in
  `electron/src/confirmHold.js`) so a server-issued hold token can be added in one place.
