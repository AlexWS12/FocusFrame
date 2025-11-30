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
- `Ctrl+Alt+S` – Start monitoring
- `Ctrl+Alt+Q` – Stop monitoring & generate report

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