import cv2
import time
import threading
import datetime
import os
import argparse
import pygetwindow as gw
from pynput import keyboard
import webbrowser
from ultralytics import YOLO


# CONFIGURATION & DEFAULTS
APP_NAME = "FocusFrame"
REPORT_FILE = "focus_frame_report.html"
MODEL_NAME = 'yolov8n.pt'
CONF_THRESHOLD = 0.4

# Presence smoothing defaults
PRESENCE_SCORE_MAX = 5
PRESENCE_THRESHOLD = 2
SCORE_INCREMENT = 1
SCORE_DECREMENT = 1
DECREMENT_INTERVAL = 0.25  # seconds between score decrements
MOTION_PIXEL_THRESHOLD = 1500
ABSENCE_TIME = 5.0  # seconds before marking away

DISTRACTION_KEYWORDS = [
    "steam", "game", "netflix", "youtube", "facebook", "twitter",
    "instagram", "tiktok", "twitch", "discord", "hulu", "prime video"
]

# YOLO COCO class IDs
CLASS_PERSON = 0
CLASS_PHONE = 67


class FocusFrameEngine:
    """Main monitoring engine for FocusFrame."""

    def __init__(self, motion_threshold=MOTION_PIXEL_THRESHOLD,
                 decrement_interval=DECREMENT_INTERVAL,
                 absence_time=ABSENCE_TIME,
                 presence_threshold=PRESENCE_THRESHOLD):
        """Initialize the engine with tunable parameters."""
        self.is_monitoring = False
        self.log_data = []
        self.start_time = None
        self.end_time = None

        # Tunable parameters
        self.motion_threshold = motion_threshold
        self.decrement_interval = decrement_interval
        self.absence_time = absence_time
        self.presence_threshold = presence_threshold

        # Presence smoothing state
        self.person_score = 0
        self._prev_gray = None
        self._last_person_decrement = time.time()

        # Tracking state
        self.last_person_seen = time.time()
        self.user_is_away = False

        # Load model
        print(f">>> {APP_NAME}: LOADING AI MODEL...")
        self.model = YOLO(MODEL_NAME)
        self.target_classes = [CLASS_PERSON, CLASS_PHONE]

        print("\n" + "="*50)
        print(f"   {APP_NAME} - READY")
        print("   [Ctrl] + [Alt] + [S]  -->  START")
        print("   [Ctrl] + [Alt] + [Q]  -->  STOP")
        print("="*50 + "\n")

    def get_active_window_title(self):
        """Get the currently active window title (lowercase)."""
        try:
            window = gw.getActiveWindow()
            return window.title.lower() if window else ""
        except Exception:
            return ""

    def log_event(self, source, message):
        """Log an event with colored console output."""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")

        # Color codes
        colors = {
            "Distraction": "\033[91m",  # Red
            "System": "\033[94m",       # Blue
            "Camera": "\033[92m"        # Green
        }
        color = colors.get(source, "\033[92m")
        reset = "\033[0m"

        print(f"{color}[{timestamp}] [{source}] {message}{reset}")

        self.log_data.append({
            "time": timestamp,
            "source": source,
            "message": message
        })

    def _detect_motion(self, frame):
        """Detect motion using frame differencing."""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            motion_detected = False

            if self._prev_gray is not None:
                diff = cv2.absdiff(gray, self._prev_gray)
                _, th = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
                motion_pixels = cv2.countNonZero(th)
                motion_detected = motion_pixels > self.motion_threshold

            self._prev_gray = gray
            return motion_detected
        except Exception:
            return False

    def _analyze_detections(self, results):
        """Parse YOLO results and return person/phone detection flags."""
        person_detected = False
        phone_detected = False

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                if conf > CONF_THRESHOLD:
                    if cls_id == CLASS_PERSON:
                        person_detected = True
                    elif cls_id == CLASS_PHONE:
                        phone_detected = True

        return person_detected, phone_detected

    def _update_presence_score(self, person_detected, motion_detected, current_time):
        """Update the presence smoothing score."""
        prev_score = self.person_score

        if person_detected or motion_detected:
            # Increment on detection
            self.person_score = min(self.person_score + SCORE_INCREMENT, PRESENCE_SCORE_MAX)
            self.last_person_seen = current_time
        else:
            # Decrement at configured interval
            if current_time - self._last_person_decrement >= self.decrement_interval:
                self.person_score = max(self.person_score - SCORE_DECREMENT, 0)
                self._last_person_decrement = current_time

        return prev_score

    def _handle_presence_logic(self, prev_score, current_time):
        """Handle away/return detection based on presence score."""
        # Return detection: score crosses threshold from below
        if self.user_is_away and prev_score < self.presence_threshold and self.person_score >= self.presence_threshold:
            self.log_event("Camera", "User returned")
            self.user_is_away = False

        # Away detection: score is zero and timeout elapsed
        if (not self.user_is_away) and (self.person_score <= 0) and (current_time - self.last_person_seen >= self.absence_time):
            self.user_is_away = True
            self.log_event("Distraction", "User Away from Desk")

    def _handle_phone_detection(self, phone_detected, current_time, last_log_time):
        """Handle phone detection logging (rate-limited)."""
        if phone_detected:
            self.last_person_seen = current_time
            self.user_is_away = False

            if current_time - last_log_time > 2.0:
                self.log_event("Distraction", "Cell Phone Detected")
                return current_time

        return last_log_time

    def _handle_screen_distraction(self, current_time, last_log_time):
        """Check active window against distraction keywords."""
        active_window = self.get_active_window_title()

        for keyword in DISTRACTION_KEYWORDS:
            if keyword in active_window:
                if current_time - last_log_time > 2.0:
                    self.log_event("Distraction", f"App: {active_window}")
                    return current_time

        return last_log_time

    def monitor_loop(self):
        """Main monitoring loop."""
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("ERROR: Could not open camera (Index 0).")
            return

        self.log_event("System", "Camera Active. Monitoring started.")
        last_log_time = 0

        while self.is_monitoring:
            ret, frame = cap.read()
            if not ret:
                break

            current_time = time.time()

            # 1. Detect motion
            motion_detected = self._detect_motion(frame)

            # 2. Run YOLO inference
            results = self.model(frame, verbose=False, classes=self.target_classes)

            # 3. Display annotated frame
            try:
                annotated_frame = results[0].plot()
                cv2.imshow("FocusFrame Vision", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    pass
            except Exception:
                pass

            # 4. Parse detections
            person_detected, phone_detected = self._analyze_detections(results)

            # 5. Update presence score
            prev_score = self._update_presence_score(person_detected, motion_detected, current_time)

            # 6. Handle presence logic (away/return)
            self._handle_presence_logic(prev_score, current_time)

            # 7. Handle phone detection
            last_log_time = self._handle_phone_detection(phone_detected, current_time, last_log_time)

            # 8. Handle screen distractions
            last_log_time = self._handle_screen_distraction(current_time, last_log_time)

        cap.release()
        cv2.destroyAllWindows()
        self.log_event("System", "Monitoring stopped.")

    def start_session(self):
        """Start a monitoring session in a background thread."""
        if self.is_monitoring:
            return

        self.is_monitoring = True
        self.start_time = datetime.datetime.now()
        self.log_data = []
        self.person_score = 0
        self.user_is_away = False
        self.last_person_seen = time.time()

        t = threading.Thread(target=self.monitor_loop)
        t.daemon = True
        t.start()

    def stop_session(self):
        """Stop the monitoring session and generate a report."""
        if not self.is_monitoring:
            return

        self.is_monitoring = False
        self.end_time = datetime.datetime.now()
        time.sleep(1)
        self.generate_report()

    def generate_report(self):
        """Generate and open an HTML report of the session."""
        if not self.start_time:
            return

        if not self.end_time:
            self.end_time = datetime.datetime.now()

        duration = self.end_time - self.start_time

        html = f"""
        <html>
        <head>
            <title>FocusFrame Report</title>
            <style>
                body {{ font-family: monospace; padding: 20px; background: #f5f5f5; }}
                h1 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; background: white; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #4CAF50; color: white; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .distraction {{ color: red; font-weight: bold; }}
                .camera {{ color: green; }}
                .system {{ color: blue; }}
            </style>
        </head>
        <body>
            <h1>FocusFrame Session Report</h1>
            <p><strong>Duration:</strong> {str(duration).split('.')[0]}</p>
            <table>
                <tr><th>Time</th><th>Type</th><th>Message</th></tr>
        """

        for entry in self.log_data:
            source = entry['source']
            cls = source.lower()
            html += f"<tr class='{cls}'><td>{entry['time']}</td><td>{source}</td><td>{entry['message']}</td></tr>"

        html += """
            </table>
        </body>
        </html>
        """

        with open(REPORT_FILE, "w", encoding="utf-8") as f:
            f.write(html)

        webbrowser.open('file://' + os.path.abspath(REPORT_FILE))
        os._exit(0)


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="FocusFrame - Monitor focus and detect distractions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python focus_frame.py
  python focus_frame.py --motion-threshold 1200 --absence-time 3
  python focus_frame.py --presence-threshold 1 --decrement-interval 0.2
        """
    )

    parser.add_argument('--motion-threshold', type=int, default=MOTION_PIXEL_THRESHOLD,
                        help=f'Motion pixel threshold (default: {MOTION_PIXEL_THRESHOLD})')
    parser.add_argument('--decrement-interval', type=float, default=DECREMENT_INTERVAL,
                        help=f'Seconds between score decrements (default: {DECREMENT_INTERVAL})')
    parser.add_argument('--absence-time', type=float, default=ABSENCE_TIME,
                        help=f'Seconds before marking away (default: {ABSENCE_TIME})')
    parser.add_argument('--presence-threshold', type=int, default=PRESENCE_THRESHOLD,
                        help=f'Score threshold to consider present (default: {PRESENCE_THRESHOLD})')

    args = parser.parse_args()

    # Create engine with CLI parameters
    app = FocusFrameEngine(
        motion_threshold=args.motion_threshold,
        decrement_interval=args.decrement_interval,
        absence_time=args.absence_time,
        presence_threshold=args.presence_threshold
    )

    # Start hotkey listener
    with keyboard.GlobalHotKeys({
        '<ctrl>+<alt>+s': app.start_session,
        '<ctrl>+<alt>+q': app.stop_session
    }) as h:
        h.join()


if __name__ == "__main__":
    main()
