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


# CONFIGURATION

class Config:
    APP_NAME = "FocusFrame"
    REPORT_FILE = "focus_frame_report.html"
    MODEL_NAME = 'yolov8n.pt'
    CONF_THRESHOLD = 0.4
    
    # Presence detection
    PRESENCE_SCORE_MAX = 5
    PRESENCE_THRESHOLD = 2
    SCORE_INCREMENT = 1
    SCORE_DECREMENT = 1
    DECREMENT_INTERVAL = 0.25
    MOTION_PIXEL_THRESHOLD = 1500
    ABSENCE_TIME = 5.0
    
    # Detection classes
    CLASS_PERSON = 0
    CLASS_PHONE = 67
    
    # Distraction keywords
    DISTRACTION_KEYWORDS = [
        "steam", "game", "netflix", "youtube", "facebook", "twitter",
        "instagram", "tiktok", "twitch", "discord", "hulu", "prime video"
    ]


# DETECTION & MONITORING

class PresenceDetector:    
    def __init__(self, config):
        self.config = config
        self.score = config.PRESENCE_SCORE_MAX
        self.prev_gray = None
        self.last_decrement = time.time()
        self.last_seen = time.time()
        self.is_away = False
        
    def reset(self):
        self.score = self.config.PRESENCE_SCORE_MAX
        self.prev_gray = None
        self.last_decrement = time.time()
        self.last_seen = time.time()
        self.is_away = False
    
    def detect_motion(self, frame):
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            motion_detected = False
            
            if self.prev_gray is not None:
                diff = cv2.absdiff(gray, self.prev_gray)
                _, th = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
                motion_pixels = cv2.countNonZero(th)
                motion_detected = motion_pixels > self.config.MOTION_PIXEL_THRESHOLD
            
            self.prev_gray = gray
            return motion_detected
        except Exception:
            return False
    
    def update_score(self, person_detected, motion_detected, current_time):
        prev_score = self.score
        
        if person_detected or motion_detected:
            self.score = min(self.score + self.config.SCORE_INCREMENT, 
                           self.config.PRESENCE_SCORE_MAX)
            self.last_seen = current_time
        else:
            if current_time - self.last_decrement >= self.config.DECREMENT_INTERVAL:
                self.score = max(self.score - self.config.SCORE_DECREMENT, 0)
                self.last_decrement = current_time
        
        return prev_score
    
    def check_presence_change(self, prev_score, current_time):
        # Return detection
        if self.is_away and prev_score < self.config.PRESENCE_THRESHOLD and self.score >= self.config.PRESENCE_THRESHOLD:
            self.is_away = False
            return "returned"
        
        # Away detection
        if (not self.is_away) and (self.score <= 0) and (current_time - self.last_seen >= self.config.ABSENCE_TIME):
            self.is_away = True
            return "away"
        
        return None


class YOLODetector:
    def __init__(self, model_path, target_classes, confidence_threshold):
        print(f">>> {Config.APP_NAME}: LOADING AI MODEL...")
        self.model = YOLO(model_path)
        self.target_classes = target_classes
        self.confidence_threshold = confidence_threshold
    
    def analyze(self, frame):
        results = self.model(frame, verbose=False, classes=self.target_classes)
        person_detected = False
        phone_detected = False
        
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                if conf > self.confidence_threshold:
                    if cls_id == Config.CLASS_PERSON:
                        person_detected = True
                    elif cls_id == Config.CLASS_PHONE:
                        phone_detected = True
        
        return results, person_detected, phone_detected


class DistractionMonitor:    
    def __init__(self, keywords):
        self.keywords = keywords
    
    def get_active_window_title(self):
        try:
            window = gw.getActiveWindow()
            return window.title.lower() if window else ""
        except Exception:
            return ""
    
    def check_distractions(self):
        active_window = self.get_active_window_title()
        for keyword in self.keywords:
            if keyword in active_window:
                return active_window
        return None


# UI & NOTIFICATIONS

class PhonePopup:
    def __init__(self):
        self.active = False
    
    def show(self, on_dismiss_callback=None):
        if self.active:
            return
        
        self.active = True
        
        def run_popup():
            root = tk.Tk()
            root.title("PHONE DETECTED - FocusFrame")
            root.attributes('-topmost', True)
            root.protocol("WM_DELETE_WINDOW", lambda: None)
            
            # Center window
            window_width, window_height = 450, 200
            x = (root.winfo_screenwidth() // 2) - (window_width // 2)
            y = (root.winfo_screenheight() // 2) - (window_height // 2)
            root.geometry(f"{window_width}x{window_height}+{x}+{y}")
            root.resizable(False, False)
            root.configure(bg='#2d2d2d')
            
            # Warning
            tk.Label(root, text="CELL PHONE DETECTED!", font=('Arial', 16, 'bold'),
                    fg='#ff4444', bg='#2d2d2d').pack(pady=(20, 10))
            
            # Instructions
            tk.Label(root, text='Type "Im back at working" to continue:',
                    font=('Arial', 11), fg='white', bg='#2d2d2d').pack(pady=(5, 10))
            
            # Entry
            entry_var = tk.StringVar()
            entry = ttk.Entry(root, textvariable=entry_var, font=('Arial', 12), width=30)
            entry.pack(pady=10)
            entry.focus_set()
            
            # Status
            status_label = tk.Label(root, text="", font=('Arial', 10),
                                   fg='#ff6666', bg='#2d2d2d')
            status_label.pack(pady=5)
            
            def check_and_close():
                if entry_var.get().strip().lower() == "im back at working":
                    self.active = False
                    if on_dismiss_callback:
                        on_dismiss_callback()
                    root.destroy()
            
            def on_enter(event):
                if entry_var.get().strip().lower() == "im back at working":
                    check_and_close()
                else:
                    status_label.config(text="Incorrect! Type exactly: Im back at working")
                    entry_var.set("")
            
            entry_var.trace('w', lambda *args: check_and_close())
            entry.bind('<Return>', on_enter)
            
            root.mainloop()
        
        threading.Thread(target=run_popup, daemon=True).start()


class EventLogger:    
    COLORS = {
        "Distraction": "\033[91m",  # Red
        "System": "\033[94m",       # Blue
        "Camera": "\033[92m"        # Green
    }
    RESET = "\033[0m"
    
    def __init__(self):
        self.log_data = []
    
    def log(self, source, message):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        color = self.COLORS.get(source, self.COLORS["Camera"])
        
        print(f"{color}[{timestamp}] [{source}] {message}{self.RESET}")
        
        self.log_data.append({
            "time": timestamp,
            "source": source,
            "message": message
        })
    
    def clear(self):
        self.log_data = []


class ReportGenerator:    
    @staticmethod
    def generate(log_data, start_time, end_time, output_file):

        if not start_time:
            return
        
        if not end_time:
            end_time = datetime.datetime.now()
        
        duration = end_time - start_time
        
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
        
        for entry in log_data:
            source = entry['source']
            cls = source.lower()
            html += f"<tr class='{cls}'><td>{entry['time']}</td><td>{source}</td><td>{entry['message']}</td></tr>"
        
        html += """
            </table>
        </body>
        </html>
        """
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html)
        
        webbrowser.open('file://' + os.path.abspath(output_file))


# MAIN ENGINE

class FocusFrameEngine:    
    def __init__(self, config=None):
        self.config = config or Config()
        self.is_monitoring = False
        self.start_time = None
        self.end_time = None
        
        # Components
        self.presence_detector = PresenceDetector(self.config)
        self.yolo_detector = YOLODetector(
            self.config.MODEL_NAME,
            [self.config.CLASS_PERSON, self.config.CLASS_PHONE],
            self.config.CONF_THRESHOLD
        )
        self.distraction_monitor = DistractionMonitor(self.config.DISTRACTION_KEYWORDS)
        self.phone_popup = PhonePopup()
        self.logger = EventLogger()
        
        # Pomodoro settings
        self.pomodoro_enabled = False
        self.pomodoro_work_min = 25
        self.pomodoro_break_min = 5
        self.pomodoro_cycles = 4
    
    def print_banner(self):
        print("\n" + "="*60)
        print(f"   {self.config.APP_NAME} - READY")
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
    
    def start_session(self):
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.start_time = datetime.datetime.now()
        self.logger.clear()
        self.presence_detector.reset()
        
        if self.pomodoro_enabled:
            threading.Thread(target=self._run_pomodoro, daemon=True).start()
        else:
            threading.Thread(target=self._run_monitoring, daemon=True).start()
    
    def stop_session(self):
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        self.end_time = datetime.datetime.now()
        time.sleep(1)
        self._generate_report()
    
    def _run_monitoring(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("ERROR: Could not open camera (Index 0).")
            return
        
        self.logger.log("System", "Camera Active. Monitoring started.")
        last_log_time = 0
        
        while self.is_monitoring:
            ret, frame = cap.read()
            if not ret:
                break
            
            current_time = time.time()
            
            # Detect motion
            motion_detected = self.presence_detector.detect_motion(frame)
            
            # Run YOLO detection
            results, person_detected, phone_detected = self.yolo_detector.analyze(frame)
            
            # Display frame
            self._display_frame(results)
            
            # Update presence
            prev_score = self.presence_detector.update_score(person_detected, motion_detected, current_time)
            
            # Check presence change
            change = self.presence_detector.check_presence_change(prev_score, current_time)
            if change == "returned":
                self.logger.log("Camera", "User returned")
            elif change == "away":
                self.logger.log("Distraction", "User Away from Desk")
            
            # Handle phone detection
            if phone_detected and current_time - last_log_time > 2.0:
                self.logger.log("Distraction", "Cell Phone Detected")
                self.phone_popup.show()
                last_log_time = current_time
            
            # Check screen distractions
            distraction = self.distraction_monitor.check_distractions()
            if distraction and current_time - last_log_time > 2.0:
                self.logger.log("Distraction", f"App: {distraction}")
                last_log_time = current_time
        
        cap.release()
        cv2.destroyAllWindows()
        self.logger.log("System", "Monitoring stopped.")
    
    def _run_pomodoro(self):
        for cycle in range(1, self.pomodoro_cycles + 1):
            if not self.is_monitoring:
                break
            
            self.logger.log("System", f"Pomodoro Cycle {cycle}/{self.pomodoro_cycles} - WORK ({self.pomodoro_work_min} min)")
            print(f"\n[Pomodoro] Cycle {cycle}/{self.pomodoro_cycles} - FOCUS TIME ({self.pomodoro_work_min} min)")
            
            # Work period
            self._run_work_period(self.pomodoro_work_min * 60)
            
            if not self.is_monitoring:
                break
            
            # Break period (except after last cycle)
            if cycle < self.pomodoro_cycles:
                self._run_break_period(cycle, self.pomodoro_break_min * 60)
        
        if self.is_monitoring:
            self.logger.log("System", "Pomodoro session completed!")
            print("\n[Pomodoro] Session completed! Great work!")
            self.is_monitoring = False
            self.end_time = datetime.datetime.now()
            self._generate_report()
    
    def _run_work_period(self, duration_seconds):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("ERROR: Could not open camera.")
            return
        
        work_remaining = duration_seconds
        last_frame_time = time.time()
        timer_paused = False
        last_log_time = 0
        
        while self.is_monitoring and work_remaining > 0:
            ret, frame = cap.read()
            if not ret:
                break
            
            current_time = time.time()
            frame_delta = current_time - last_frame_time
            
            # Detect
            motion_detected = self.presence_detector.detect_motion(frame)
            results, person_detected, phone_detected = self.yolo_detector.analyze(frame)
            
            # Update presence
            prev_score = self.presence_detector.update_score(person_detected, motion_detected, current_time)
            change = self.presence_detector.check_presence_change(prev_score, current_time)
            
            if change == "returned":
                self.logger.log("Camera", "User returned")
            elif change == "away":
                self.logger.log("Distraction", "User Away from Desk")
            
            # Phone detection
            if phone_detected and current_time - last_log_time > 2.0:
                self.logger.log("Distraction", "Cell Phone Detected")
                self.phone_popup.show()
                last_log_time = current_time
            
            # Screen distractions
            distraction = self.distraction_monitor.check_distractions()
            if distraction and current_time - last_log_time > 2.0:
                self.logger.log("Distraction", f"App: {distraction}")
                last_log_time = current_time
            
            # Timer pause logic
            should_pause = self.presence_detector.is_away or phone_detected or self.phone_popup.active
            
            if not should_pause:
                work_remaining -= frame_delta
                if timer_paused:
                    print("\n[Timer RESUMED]")
                    timer_paused = False
            else:
                if not timer_paused:
                    print("\n[Timer PAUSED - away/phone detected]")
                    timer_paused = True
            
            # Display with timer
            self._display_frame_with_timer(results, int(max(0, work_remaining)), timer_paused)
            
            last_frame_time = current_time
        
        cap.release()
        cv2.destroyAllWindows()
    
    def _run_break_period(self, cycle, duration_seconds):
        self.logger.log("System", f"Pomodoro Cycle {cycle} - BREAK ({self.pomodoro_break_min} min)")
        print(f"\n[Pomodoro] BREAK TIME ({self.pomodoro_break_min} min) - Relax!")
        
        break_end = time.time() + duration_seconds
        while self.is_monitoring and time.time() < break_end:
            remaining = int(break_end - time.time())
            mins, secs = divmod(remaining, 60)
            print(f"\r   Break remaining: {mins:02d}:{secs:02d}  ", end="", flush=True)
            time.sleep(1)
        print()
    
    def _display_frame(self, results):
        try:
            annotated_frame = results[0].plot()
            cv2.imshow("FocusFrame Vision", annotated_frame)
            cv2.waitKey(1)
        except Exception:
            pass
    
    def _display_frame_with_timer(self, results, remaining_seconds, paused):
        try:
            annotated_frame = results[0].plot()
            mins, secs = divmod(remaining_seconds, 60)
            status = "PAUSED" if paused else "FOCUS"
            color = (0, 165, 255) if paused else (0, 255, 0)
            cv2.putText(annotated_frame, f"{status}: {mins:02d}:{secs:02d}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.imshow("FocusFrame Vision", annotated_frame)
            cv2.waitKey(1)
        except Exception:
            pass
    
    def _generate_report(self):
        ReportGenerator.generate(
            self.logger.log_data,
            self.start_time,
            self.end_time,
            self.config.REPORT_FILE
        )
        os._exit(0)


# CLI & MAIN

def setup_pomodoro_interactive(app):
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
        print(f"\n   Pomodoro configured: {app.pomodoro_work_min}m work / {app.pomodoro_break_min}m break x {app.pomodoro_cycles} cycles")
        print("   Press [Ctrl]+[Alt]+[Enter] to start Pomodoro session!")
    except ValueError:
        print("   Invalid input. Pomodoro disabled.")
        app.pomodoro_enabled = False
    print("-"*50 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="FocusFrame - Monitor focus and detect distractions",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--motion-threshold', type=int, default=Config.MOTION_PIXEL_THRESHOLD)
    parser.add_argument('--decrement-interval', type=float, default=Config.DECREMENT_INTERVAL)
    parser.add_argument('--absence-time', type=float, default=Config.ABSENCE_TIME)
    parser.add_argument('--presence-threshold', type=int, default=Config.PRESENCE_THRESHOLD)
    
    args = parser.parse_args()
    
    # Apply CLI configuration
    config = Config()
    config.MOTION_PIXEL_THRESHOLD = args.motion_threshold
    config.DECREMENT_INTERVAL = args.decrement_interval
    config.ABSENCE_TIME = args.absence_time
    config.PRESENCE_THRESHOLD = args.presence_threshold
    
    # Create engine
    app = FocusFrameEngine(config)
    app.print_banner()
    
    # 'p' command
    def input_listener():
        while True:
            try:
                cmd = input().strip().lower()
                if cmd == 'p':
                    setup_pomodoro_interactive(app)
                    app.print_banner()
                elif cmd in ('help', 'h'):
                    app.print_banner()
            except EOFError:
                break
    
    threading.Thread(target=input_listener, daemon=True).start()
    
    # Hotkey listener
    with keyboard.GlobalHotKeys({
        '<ctrl>+<alt>+<enter>': app.start_session,
        '<ctrl>+<alt>+<backspace>': app.stop_session
    }) as h:
        h.join()


if __name__ == "__main__":
    main()
