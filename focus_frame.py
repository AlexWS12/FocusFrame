import cv2
import time
import threading
import datetime
import os
import argparse
import tkinter as tk
from tkinter import ttk
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
        self.person_score = PRESENCE_SCORE_MAX  # Start present
        self._prev_gray = None
        self._last_person_decrement = time.time()

        # Tracking state
        self.last_person_seen = time.time()
        self.user_is_away = False

        # Pomodoro settings (configured via interactive menu)
        self.pomodoro_enabled = False
        self.pomodoro_work_min = 25
        self.pomodoro_break_min = 5
        self.pomodoro_cycles = 4
        self.pomodoro_current_cycle = 0
        
        # Phone popup tracking
        self._popup_active = False

        # Load model
        print(f">>> {APP_NAME}: LOADING AI MODEL...")
        self.model = YOLO(MODEL_NAME)
        self.target_classes = [CLASS_PERSON, CLASS_PHONE]

    def print_ready_banner(self):
        """Print the ready banner with hotkeys and Pomodoro info."""
        print("\n" + "="*60)
        print(f"   {APP_NAME} - READY")
        print("="*60)
        print("   HOTKEYS:")
        print("   [Ctrl] + [Alt] + [Enter]      -->  START")
        print("   [Ctrl] + [Alt] + [Backspace]  -->  STOP")
        print("-"*60)
        print("   POMODORO MODE (optional):")
        print("   Type 'p' + Enter to configure Pomodoro timer")
        print("   Example: 25 min work, 5 min break, 4 cycles")
        print("="*60)
        if self.pomodoro_enabled:
            print(f"   [POMODORO ACTIVE] {self.pomodoro_work_min}m work / {self.pomodoro_break_min}m break x {self.pomodoro_cycles} cycles")
        else:
            print("   [POMODORO OFF] Press hotkey to start manual session")
        print("="*60 + "\n")

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

    def _show_phone_popup(self):
        """Show a blocking popup that requires typing 'Im back at working' to dismiss."""
        if self._popup_active:
            return
        
        self._popup_active = True
        
        def run_popup():
            root = tk.Tk()
            root.title("PHONE DETECTED - FocusFrame")
            root.attributes('-topmost', True)
            root.protocol("WM_DELETE_WINDOW", lambda: None)
            
            window_width = 450
            window_height = 200
            screen_width = root.winfo_screenwidth()
            screen_height = root.winfo_screenheight()
            x = (screen_width // 2) - (window_width // 2)
            y = (screen_height // 2) - (window_height // 2)
            root.geometry(f"{window_width}x{window_height}+{x}+{y}")
            root.resizable(False, False)
            root.configure(bg='#2d2d2d')
            
            warning_label = tk.Label(
                root,
                text="CELL PHONE DETECTED!",
                font=('Arial', 16, 'bold'),
                fg='#ff4444',
                bg='#2d2d2d'
            )
            warning_label.pack(pady=(20, 10))
            
            instruction_label = tk.Label(
                root,
                text='Type "Im back at working" to continue:',
                font=('Arial', 11),
                fg='white',
                bg='#2d2d2d'
            )
            instruction_label.pack(pady=(5, 10))
            
            entry_var = tk.StringVar()
            entry = ttk.Entry(root, textvariable=entry_var, font=('Arial', 12), width=30)
            entry.pack(pady=10)
            entry.focus_set()
            
            status_label = tk.Label(
                root,
                text="",
                font=('Arial', 10),
                fg='#ff6666',
                bg='#2d2d2d'
            )
            status_label.pack(pady=5)
            
            def check_input(*args):
                if entry_var.get().strip().lower() == "im back at working":
                    self._popup_active = False
                    root.destroy()
            
            def on_enter(event):
                if entry_var.get().strip().lower() == "im back at working":
                    self._popup_active = False
                    root.destroy()
                else:
                    status_label.config(text="Incorrect! Type exactly: Im back at working")
                    entry_var.set("")
            
            entry_var.trace('w', check_input)
            entry.bind('<Return>', on_enter)
            
            root.mainloop()
        
        popup_thread = threading.Thread(target=run_popup, daemon=True)
        popup_thread.start()

    def _handle_phone_detection(self, phone_detected, current_time, last_log_time):
        """Handle phone detection logging (rate-limited) with popup."""
        if phone_detected:
            self.last_person_seen = current_time
            self.user_is_away = False

            if current_time - last_log_time > 2.0:
                self.log_event("Distraction", "Cell Phone Detected")
                self._show_phone_popup()
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

            #  Detect motion
            motion_detected = self._detect_motion(frame)

            #  Run YOLO inference
            results = self.model(frame, verbose=False, classes=self.target_classes)

            #  Display annotated frame
            try:
                annotated_frame = results[0].plot()
                cv2.imshow("FocusFrame Vision", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    pass
            except Exception:
                pass

            #  Parse detections
            person_detected, phone_detected = self._analyze_detections(results)

            #  Update presence score
            prev_score = self._update_presence_score(person_detected, motion_detected, current_time)

            #  Handle presence logic (away/return)
            self._handle_presence_logic(prev_score, current_time)

            #  Handle phone detection
            last_log_time = self._handle_phone_detection(phone_detected, current_time, last_log_time)

            #  Handle screen distractions
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
        self.person_score = PRESENCE_SCORE_MAX  # Start present
        self.user_is_away = False
        self.last_person_seen = time.time()
        self._prev_gray = None  # Reset motion detection

        if self.pomodoro_enabled:
            # Start Pomodoro-managed session
            t = threading.Thread(target=self._pomodoro_loop)
            t.daemon = True
            t.start()
        else:
            # Start regular monitoring
            t = threading.Thread(target=self.monitor_loop)
            t.daemon = True
            t.start()

    def _pomodoro_loop(self):
        """Run Pomodoro cycles with work/break intervals."""
        for cycle in range(1, self.pomodoro_cycles + 1):
            if not self.is_monitoring:
                break

            self.pomodoro_current_cycle = cycle
            self.log_event("System", f"Pomodoro Cycle {cycle}/{self.pomodoro_cycles} - WORK ({self.pomodoro_work_min} min)")
            print(f"\n[Pomodoro] Cycle {cycle}/{self.pomodoro_cycles} - FOCUS TIME ({self.pomodoro_work_min} min)")

            # Work period with monitoring and timer pausing
            work_remaining = self.pomodoro_work_min * 60  # seconds
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("ERROR: Could not open camera.")
                return

            last_log_time = 0
            last_frame_time = time.time()
            timer_paused = False

            while self.is_monitoring and work_remaining > 0:
                ret, frame = cap.read()
                if not ret:
                    break

                current_time = time.time()
                motion_detected = self._detect_motion(frame)
                results = self.model(frame, verbose=False, classes=self.target_classes)

                try:
                    annotated_frame = results[0].plot()
                    remaining = int(max(0, work_remaining))
                    mins, secs = divmod(remaining, 60)
                    status = "PAUSED" if timer_paused else "FOCUS"
                    color = (0, 165, 255) if timer_paused else (0, 255, 0)
                    cv2.putText(annotated_frame, f"{status}: {mins:02d}:{secs:02d}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    cv2.imshow("FocusFrame Vision", annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        pass
                except Exception:
                    pass

                person_detected, phone_detected = self._analyze_detections(results)
                prev_score = self._update_presence_score(person_detected, motion_detected, current_time)
                self._handle_presence_logic(prev_score, current_time)
                last_log_time = self._handle_phone_detection(phone_detected, current_time, last_log_time)
                last_log_time = self._handle_screen_distraction(current_time, last_log_time)

                # Pause timer if away or phone detected (only during work, not break)
                frame_delta = current_time - last_frame_time
                should_pause = self.user_is_away or phone_detected or self._popup_active
                
                if not should_pause:
                    work_remaining -= frame_delta
                    if timer_paused:
                        print("\n[Timer RESUMED]")
                        timer_paused = False
                else:
                    if not timer_paused:
                        print("\n[Timer PAUSED - away/phone detected]")
                        timer_paused = True
                
                last_frame_time = current_time

            cap.release()
            cv2.destroyAllWindows()

            if not self.is_monitoring:
                break

            # Break period (except after last cycle)
            if cycle < self.pomodoro_cycles:
                self.log_event("System", f"Pomodoro Cycle {cycle} - BREAK ({self.pomodoro_break_min} min)")
                print(f"\n[Pomodoro] BREAK TIME ({self.pomodoro_break_min} min) - Relax!")

                break_end = time.time() + (self.pomodoro_break_min * 60)
                while self.is_monitoring and time.time() < break_end:
                    remaining = int(break_end - time.time())
                    mins, secs = divmod(remaining, 60)
                    print(f"\r   Break remaining: {mins:02d}:{secs:02d}  ", end="", flush=True)
                    time.sleep(1)
                print()  # newline after break countdown

        if self.is_monitoring:
            self.log_event("System", "Pomodoro session completed!")
            print("\n[Pomodoro] Session completed! Great work!")
            self.is_monitoring = False
            self.end_time = datetime.datetime.now()
            self.generate_report()

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

    # Interactive Pomodoro setup
    def setup_pomodoro():
        """Interactive CLI to configure Pomodoro settings."""
        print("\n" + "-"*50)
        print("   POMODORO SETUP")
        print("-"*50)
        try:
            work = input("   Work duration (minutes) [25]: ").strip()
            app.pomodoro_work_min = int(work) if work else 25

            brk = input("   Break duration (minutes) [5]: ").strip()
            app.pomodoro_break_min = int(brk) if brk else 5

            cycles = input("   Number of cycles [4]: ").strip()
            app.pomodoro_cycles = int(cycles) if cycles else 4

            app.pomodoro_enabled = True
            print(f"\n   ✓ Pomodoro configured: {app.pomodoro_work_min}m work / {app.pomodoro_break_min}m break x {app.pomodoro_cycles} cycles")
            print("   Press [Ctrl]+[Alt]+[Enter] to start Pomodoro session!")
        except ValueError:
            print("   Invalid input. Pomodoro disabled.")
            app.pomodoro_enabled = False
        print("-"*50 + "\n")

    # Print ready banner
    app.print_ready_banner()

    # Start input listener for 'p' command in a thread
    def input_listener():
        while True:
            try:
                cmd = input().strip().lower()
                if cmd == 'p':
                    setup_pomodoro()
                    app.print_ready_banner()
                elif cmd == 'help' or cmd == 'h':
                    app.print_ready_banner()
            except EOFError:
                break

    input_thread = threading.Thread(target=input_listener, daemon=True)
    input_thread.start()

    # Start hotkey listener
    with keyboard.GlobalHotKeys({
        '<ctrl>+<alt>+<enter>': app.start_session,
        '<ctrl>+<alt>+<backspace>': app.stop_session
    }) as h:
        h.join()


if __name__ == "__main__":
    main()
