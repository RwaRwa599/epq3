# Continuity — new2

> Shared ground truth for project state across all agents and sessions.
> Update at the end of every session. Never delete — only archive (see `REVIEW.md`).
>
> Each fact carries a metadata footer in an HTML comment, maintained by the review
> ritual — invisible when rendered, read/written by agents:
> `<!-- id: kebab-id | created: YYYY-MM-DD | last_used: YYYY-MM-DD | uses: N | tier: active -->`
> See `.agent/schema.md` for the fields and `memory/decay-policy.md` for the windows.

---

## Project State

- **project:** new2
- **status:** Greenfield initialization — ready for stack and feature definition
- **last_enabled:** 2026-08-14
- **last_session:** 2026-08-14 | agent: Cursor (2026-08-14-023004)
- **last_review:** (none yet)
- **last_invariant_check:** (none yet)
- **repo:** ~/new2

## Stack & Tools

> Canonical live home for the current stack — language version, dependencies, tool
> versions. `instructions.md` keeps only a high-level descriptor and points here.

(none yet — greenfield)

## Key Decisions

- Initialized with agent-memory v4.32.1 (Mode A, deep analysis)
  <!-- id: init-agent-memory-v4321 | created: 2026-08-14 | last_used: 2026-08-14 | uses: 1 | tier: working | origin: 2026-08-14-023004 -->

## Conventions

(none yet — to be established)

## Open Threads

> Mark completed items `- [x]` and leave them in place — the review sweeps them to
> the archive once older than `archive_window` sessions. Don't archive them by hand.

- [ ] (vision-bootstrap) Confirm the Vision in memory/vision.md — set the target / success criteria / non-goals; then derive the Blueprint.
  <!-- id: vision-bootstrap | created: 2026-08-14 | last_used: 2026-08-14 | uses: 1 | tier: active | origin: 2026-08-14-023004 -->
- [ ] Greenfield — no code yet: record the stack in ## Stack & Tools, coding conventions, Architectural Invariants, and seed the stack's build-output .gitignore entries when the stack lands.
  <!-- id: greenfield-seed-stack | created: 2026-08-14 | last_used: 2026-08-14 | uses: 1 | tier: active | origin: 2026-08-14-023004 -->

## User Preferences

(none recorded yet — record ONLY what the user explicitly states; never infer)

## Team / Members

(none recorded yet)
