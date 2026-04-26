# Video recording checklist

## Capture

- **Tool:** OBS Studio (recommended, free) or equivalent; record display + system audio if you add UI sounds.
- **Resolution / framerate:** 1920×1080, 60 fps.
- **Audio:** clear voice, no room noise; record a 10 s noise profile if using noise suppression.
- **Inputs:** full screen or window around terminal + browser; avoid unreadable font sizes (terminal ≥ 14 pt equivalent).

## B-roll (get each clip 8–20 s, trim in edit)

1. Terminal: GRPO job streaming logs (`reward`, `step`, `entropy` lines visible).
2. Trackio (main Space): live run dashboard, one pan across key panels.
3. HF Space: SevZero environment UI or API flow stepping through an episode.
4. HF Model card: `mist-ic/sevzero-llama3-8b-grpo` (name, base model, adapter, links).
5. Optional: one cut of `assets/reward_curve.png` full screen for a static beat (curve + annotations + learning delta).

## Edit

- **Pace:** hard cuts, no long idle holds; target under 2 minutes total.
- **Accessibility:** burn in subtitles (YouTube or editor captions export to SRT and bake-in for HF if required).
- **Overlays:** use exact lines from `VIDEO_SCRIPT.md` for on-screen text; keep contrast AA-friendly.

## Export

- **Container:** H.264 or VP9, 1080p, bitrate sufficient for screen text (avoid heavy compression artifacts on log output).
- **Thumb:** static frame = reward curve or split before/after, not a generic stock image.
