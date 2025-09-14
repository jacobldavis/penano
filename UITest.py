from audio.pen_audio import AudioMixer, P_FREQ
from ultralytics import YOLO # type: ignore
import cv2
import random
import mediapipe as mp # type: ignore
import tkinter as tk
import threading
from PIL import Image, ImageTk
from audio.pen_audio import *
from datetime import datetime

def getColours(cls_num):
    """Generate unique colors for each class ID"""
    random.seed(cls_num)
    return tuple(random.randint(0, 255) for _ in range(3))

def is_point_in_rectangle(point, rect):
    """Check if a point is inside a rectangle"""
    x, y = point
    x1, y1, x2, y2 = rect[:4]
    return x1 <= x <= x2 and y1 <= y <= y2

def get_fingertip_positions(hand_landmarks):
    """Extract fingertip positions from hand landmarks"""
    # MediaPipe hand landmark indices for fingertips
    fingertip_ids = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
    fingertips = []
    
    for tip_id in fingertip_ids:
        landmark = hand_landmarks.landmark[tip_id]
        fingertips.append((landmark.x, landmark.y))
    
    return fingertips

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

class PenanoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PENANO")
        self.root.configure(bg='#2C2F33')  # Dark theme background
        
        # Initialize state variables first
        self.running = True
        self.rectangles = []
        self.previous_touched_notes = []

        # Create main frames
        self.main_frame = tk.Frame(root, bg='#2C2F33')
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Video frame with border
        self.video_frame = tk.Frame(self.main_frame, bg='#23272A', bd=2, relief='solid')
        self.video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        self.video_label = tk.Label(self.video_frame, bg='#23272A')
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        # Control frame with gradient-like background
        self.control_frame = tk.Frame(self.main_frame, width=250, bg='#23272A')
        self.control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5)
        self.control_frame.pack_propagate(False)

        # Title with better styling
        title_label = tk.Label(
            self.control_frame,
            text="PENANO CONTROLS",
            bg='#7289DA',  # Discord-like blue
            fg='white',
            font=("Helvetica", 14, "bold"),
            pady=10
        )
        title_label.pack(fill='x', pady=(0, 15))

        # Camera controls with better styling
        tk.Label(
            self.control_frame,
            text="Camera Index",
            bg='#23272A',
            fg='#99AAB5',
            font=("Helvetica", 10, "bold")
        ).pack(pady=(5,0))
        
        self.camera_var = tk.StringVar(value="1")
        entry = tk.Entry(
            self.control_frame,
            textvariable=self.camera_var,
            justify='center',
            bg='#2C2F33',
            fg='white',
            insertbackground='white',
            relief='flat',
            width=10
        )
        entry.pack(pady=5)
        
        tk.Button(
            self.control_frame,
            text="Change Camera",
            command=self.change_camera,
            bg='#7289DA',
            fg='white',
            activebackground='#677BC4',
            activeforeground='white',
            font=("Helvetica", 9),
            relief='flat',
            pady=5
        ).pack(pady=(0,10))

        # Octave control with better styling
        tk.Label(
            self.control_frame,
            text="Octave",
            bg='#23272A',
            fg='#99AAB5',
            font=("Helvetica", 10, "bold")
        ).pack(pady=(15,5))
        
        self.octave_var = tk.IntVar(value=4)
        self.octave_scrollbar = tk.Scale(
            self.control_frame,
            from_=1,
            to=8,
            orient='horizontal',
            variable=self.octave_var,
            command=self.on_octave_change,
            bg='#2C2F33',
            fg='white',
            activebackground='#7289DA',
            troughcolor='#99AAB5',
            relief='flat',
            highlightthickness=0
        )
        self.octave_scrollbar.pack(fill='x', padx=20, pady=(0,15))

        # Separator
        tk.Frame(self.control_frame, height=2, bg='#99AAB5').pack(fill='x', pady=15, padx=20)

        # Action buttons with better styling
        button_frame = tk.Frame(self.control_frame, bg='#23272A')
        button_frame.pack(pady=10)
        
        for text, command in [
            ("Classify (W)", self.toggle_detection),
            ("Record (R)", self.toggle_recording),
            ("Quit (Q)", self.on_closing)
        ]:
            tk.Button(
                button_frame,
                text=text,
                command=command,
                bg='#7289DA',
                fg='white',
                activebackground='#677BC4',
                activeforeground='white',
                font=("Helvetica", 9),
                relief='flat',
                pady=8,
                width=15
            ).pack(pady=5)

        # Initialize recorder before mixer
        self.recorder = AudioRecorder(sample_rate=44100)
        self.mixer = RecordableAudioMixer(sample_rate=44100, block_size=512)
        self.mixer.set_recorder(self.recorder)

        # Add record hotkey
        self.root.bind('r', lambda e: self.toggle_recording())

        # Initialize components
        self.model = YOLO("Train_Yolo/my_model.pt")
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )

        # Video capture
        self.change_camera()

        # Start video thread
        self.video_thread = threading.Thread(target=self.process_video)
        self.video_thread.daemon = True
        self.video_thread.start()

        # Bind quit event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.bind('q', lambda e: self.on_closing())
        self.root.bind('w', lambda e: self.toggle_detection())

    def change_camera(self):
        try:
            if hasattr(self, 'cap'):
                self.cap.release()
            idx = int(self.camera_var.get())
            self.cap = cv2.VideoCapture(idx)
        except:
            print("Error changing camera")

    def on_octave_change(self, value):
        """Handle octave scrollbar changes"""
        if hasattr(self.mixer, 'set_octave'):
            self.mixer.set_octave(int(value))

    def toggle_detection(self):
        # Reset rectangles and detect new ones
        self.rectangles = []
        ret, frame = self.cap.read()
        if ret:
            results = self.model.track(frame, stream=True)
            for result in results:
                class_names = result.names
                for box in result.boxes:
                    if box.conf[0] > 0.4:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cls = int(box.cls[0])
                        class_name = class_names[cls]
                        conf = float(box.conf[0])
                        colour = getColours(cls)
                        note = class_name
                        # Store rectangle information with note
                        self.rectangles.append((x1, y1, x2, y2, colour, class_name, conf, note))

    def toggle_recording(self):
        """Toggle recording on/off"""
        if not self.recorder.is_recording():
            self.recorder.start_recording()
        else:
            self.recorder.stop_recording()
            filename = self.recorder.save_to_mp3("recording.mp3")
            if filename:
                print(f"Recording saved as: {filename}")

    def process_video(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hand_results = self.hands.process(rgb_frame)
            current_touches = set()
            
            # Display status text
            cv2.putText(frame, f"Keys: {len(self.rectangles)}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            cv2.putText(frame, f"Octave: {self.octave_var.get()}", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            
            # Display recording status like in main.py
            if self.recorder.is_recording():
                cv2.putText(frame, "REC", 
                           (frame.shape[1] - 80, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    fingertips = get_fingertip_positions(hand_landmarks)
                    
                    for norm_x, norm_y in fingertips:
                        pixel_x = int(norm_x * frame.shape[1])
                        pixel_y = int(norm_y * frame.shape[0])
                        cv2.circle(frame, (pixel_x, pixel_y), 8, (0, 255, 0), -1)
                        
                        for rect_idx, rect in enumerate(self.rectangles):
                            x1, y1, x2, y2, colour, class_name, _, note = rect
                            if is_point_in_rectangle((pixel_x, pixel_y), rect):
                                touch_id = f"{rect_idx}_{class_name}_{note}"
                                current_touches.add(touch_id)
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 4)
            
            # Draw the stored rectangles with labels
            for rect in self.rectangles:
                x1, y1, x2, y2, colour, class_name, conf, note = rect
                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
                # Draw note name centered in rectangle
                text_size = cv2.getTextSize(note, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
                text_x = x1 + (x2 - x1 - text_size[0]) // 2
                text_y = y1 + (y2 - y1 + text_size[1]) // 2
                cv2.putText(frame, note, (text_x, text_y), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1.0, colour, 2)
            
            # Draw currently playing notes
            if self.previous_touched_notes:
                notes_text = "Playing: " + ", ".join(self.previous_touched_notes)
                cv2.putText(frame, notes_text, 
                           (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Update audio
            touched_notes = {touch_id.split('_')[-1] for touch_id in current_touches}
            if touched_notes != self.previous_touched_notes:
                self.mixer.update_notes(list(touched_notes))
                self.previous_touched_notes = touched_notes.copy()
            
            # Convert and display frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.config(image=imgtk)
            self.video_label.image = imgtk

    def on_closing(self):
        if self.recorder.is_recording():
            self.recorder.stop_recording()
            filename = self.recorder.save_to_mp3()
            if filename:
                print(f"Final recording saved as: {filename}")
        self.running = False
        if hasattr(self, 'cap'):
            self.cap.release()
        self.mixer.cleanup()
        self.hands.close()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = PenanoApp(root)
    root.mainloop()