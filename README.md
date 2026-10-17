# eye-tracker

Gaze zone and blink detection from an ordinary webcam, built as an accessibility
input method for people with limited mobility.

Commercial eye-tracking hardware costs thousands of dollars. This runs on the
camera a laptop already has.

<!-- TODO: record a 10-second screen capture running the script (look left,
     right, up, down, then blink twice), save it as media/demo.gif, and
     uncomment the line below. This matters more than the code does. -->
<!-- ![Demo](media/demo.gif) -->

## What it does

- Detects the face and iris using the MediaPipe Face Landmarker (Tasks API)
- Classifies gaze into five zones: center, left, right, up, down
- Detects blinks via eye aspect ratio
- Runs in real time on CPU, no GPU required

## What it deliberately does not do

**No pixel-level cursor control.** Gaze estimation from a consumer webcam is not
accurate enough for screen coordinates, and projects that attempt it tend to
produce something that demos badly and frustrates real users. Five coarse zones
plus a blink are enough to drive a selection interface, and they work reliably.

This is a scope decision, not a known bug.

## Install

Requires Python 3.9 or newer and a working webcam.

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

pip install -r requirements.txt
```

## Run

```bash
python eye_tracker.py
```

| Key | Action |
| --- | ------ |
| `c` | Calibrate: treat your current gaze as dead center |
| `q` | Quit |

The first run downloads the face landmark model (a few MB), so it needs an
internet connection once. After that it runs fully offline.

Calibrate once after starting, looking straight at the middle of the screen.
Everything is measured relative to that point.

## How it works

**Blink detection** uses the eye aspect ratio: the vertical opening of the eye
divided by its horizontal width. The ratio collapses when the eyelid closes.
Because it is a ratio, it is unaffected by how close the user sits to the camera
or by lighting and skin tone.

**Gaze zones** use the iris center position relative to the eye corners,
normalized by eye width for the same reason. The result is smoothed across
several frames to reduce jitter, then thresholded into zones.

## Tuning

If detection feels wrong, adjust the constants at the top of `eye_tracker.py`:

| Constant | Raise it if | Lower it if |
| -------- | ----------- | ----------- |
| `EAR_THRESHOLD` | blinks are missed | normal eyes read as closed |
| `GAZE_MARGIN_X` | left/right triggers too easily | left/right is hard to trigger |
| `GAZE_MARGIN_Y` | up/down triggers too easily | up/down is hard to trigger |
| `SMOOTHING` | the reading jitters | the reading feels laggy |

## Roadmap

- [ ] v1: gaze zones and blink detection
- [ ] Demo GIF in this README
- [ ] v2: on-screen keyboard driven by zones and blinks
- [ ] v2: per-user calibration routine
- [ ] v2: measured accuracy, reported as numbers

## License

MIT
