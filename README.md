# FocusFrame

Author: Alex Waisman

Monitor your focus and detect digital distractions in real-time.

## Features
- **Person Detection:** Tracks when you're away from your desk using YOLO AI.
- **Phone Detection:** Alerts when a cell phone is detected.
- **App Monitoring:** Logs when you open distracting apps (Netflix, YouTube, Discord, etc.).
- **Motion Fallback:** Detects movement even if AI misses you.
- **Session Reports:** HTML report with timestamped event log.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Start the app:
```bash
python focus_frame.py
```

**Hotkeys:**
- `Ctrl+Alt+Enter` – Start monitoring
- `Ctrl+Alt+Backspace` – Stop monitoring & generate report

## Pomodoro Mode

After launching, type `p` and press Enter to configure Pomodoro:

```
>>> FocusFrame: LOADING AI MODEL...

============================================================
   FocusFrame - READY
============================================================
   HOTKEYS:
   [Ctrl] + [Alt] + [Enter]      -->  START
   [Ctrl] + [Alt] + [Backspace]  -->  STOP
------------------------------------------------------------
   POMODORO MODE (optional):
   Type 'p' + Enter to configure Pomodoro timer
   Example: 25 min work, 5 min break, 4 cycles
============================================================

p
--------------------------------------------------
   POMODORO SETUP
--------------------------------------------------
   Work duration (minutes): 25
   Break duration (minutes): 5
   Number of cycles: 4

   ✓ Pomodoro configured: 25m work / 5m break x 4 cycles
   Press [Ctrl]+[Alt]+[Enter] to start Pomodoro session!
--------------------------------------------------
```

Then press `Ctrl+Alt+Enter` to start your Pomodoro session with automatic work/break cycles.

## Tuning Parameters

```bash
python focus_frame.py --motion-threshold 1200 --absence-time 3 --presence-threshold 1
```

**Available flags:**
- `--motion-threshold` (default: 1500) – Lower = more sensitive to movement
- `--absence-time` (default: 5.0) – Seconds before marking away
- `--presence-threshold` (default: 2) – Score needed to register as present
- `--decrement-interval` (default: 0.25) – Seconds between score drops

## How It Works

1. **Presence Smoothing:** Uses a score system (0-5) that increments on detection/motion and decrements slowly.
2. **Away Detection:** Logs "User Away from Desk" when score reaches 0 and timeout elapses.
3. **Return Detection:** Logs "User returned" when score crosses the presence threshold.
4. **Event Logging:** All events (phone, apps, away/return) logged to HTML report.

## Requirements

- Python 3.8+
- OpenCV, YOLO, pynput, pygetwindow
- Webcam

See `requirements.txt` for full dependency list.

## License

MIT License