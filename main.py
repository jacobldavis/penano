from audio.pen_audio import *
from ultralytics import YOLO # type: ignore
import cv2
import random
import mediapipe as mp # type: ignore
import numpy as np
import wave
import threading
from pydub import AudioSegment
import os
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

# Initialize audio recorder and mixer
recorder = AudioRecorder(sample_rate=44100)
mixer = RecordableAudioMixer(sample_rate=44100, block_size=512)
mixer.set_recorder(recorder)

# Load YOLO model
model = YOLO("Train_Yolo/my_model.pt")

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video stream.")
    mixer.cleanup()
    exit()

# Store the rectangles to draw
rectangles = []

# Keep track of which notes were previously touched
previous_touched_notes = []

print("Controls:")
print("- Press 'w' to detect piano keys")
print("- Press 'r' to start/stop recording")
print("- Press 'q' to quit")
print("- Touch detected piano keys with your fingers to play!")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get frame dimensions
        height, width, _ = frame.shape
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('w'):
            # Reset rectangles and detect new ones
            rectangles = []
            results = model.track(frame, stream=True)
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
                        rectangles.append((x1, y1, x2, y2, colour, class_name, conf, note))
            print(f"Detected {len(rectangles)} piano keys")
        
        elif key == ord('r'):
            # Toggle recording
            if not recorder.is_recording():
                recorder.start_recording()
            else:
                recorder.stop_recording()
                filename = recorder.save_to_mp3("recording.mp3")
                if filename:
                    print(f"Recording saved as: {filename}")
        
        # Process hand detection
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_results = hands.process(rgb_frame)
        
        # Track current touches for this frame
        current_touches = set()
        
        # Draw hand landmarks and check for interactions
        if hand_results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                # Draw hand landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Get fingertip positions
                fingertips = get_fingertip_positions(hand_landmarks)
                
                # Convert normalized coordinates to pixel coordinates
                for finger_idx, (norm_x, norm_y) in enumerate(fingertips):
                    pixel_x = int(norm_x * width)
                    pixel_y = int(norm_y * height)
                    
                    # Draw fingertip points
                    cv2.circle(frame, (pixel_x, pixel_y), 8, (0, 255, 0), -1)
                    
                    # Check intersection with rectangles
                    for rect_idx, rect in enumerate(rectangles):
                        x1, y1, x2, y2, colour, class_name, conf, note = rect
                        
                        if is_point_in_rectangle((pixel_x, pixel_y), rect):
                            # Create unique identifier for this touch
                            touch_id = f"{rect_idx}_{class_name}_{note}"
                            current_touches.add(touch_id)
                            
                            # Visual feedback - make the rectangle flash and show note info
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 4)
                            cv2.putText(frame, f"PLAYING: {note}", 
                                      (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                            cv2.putText(frame, f"{class_name}", 
                                      (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Update the audio mixer with currently touched notes
        touched_notes = set()
        for touch_id in current_touches:
            # Extract note from touch_id (format: rect_idx_class_name_note)
            parts = touch_id.split('_')
            note = parts[-1]  
            touched_notes.add(note)
    
        # Only update audio if the touched notes have changed
        if touched_notes != previous_touched_notes:
            mixer.update_notes(list(touched_notes))
            previous_touched_notes = touched_notes.copy()
        
        # Draw the stored rectangles on the current frame
        for rect in rectangles:
            x1, y1, x2, y2, colour, class_name, conf, note = rect
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
            cv2.putText(frame, f"{class_name} ({note}) {conf:.2f}",
                       (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)
        
        # Display active notes info
        active_notes = mixer.get_active_notes()
        if active_notes:
            notes_text = "Playing: " + ", ".join(active_notes)
            cv2.putText(frame, notes_text, 
                       (10, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Display recording status
        if recorder.is_recording():
            cv2.putText(frame, "● REC", 
                       (width - 80, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Display instructions
        cv2.putText(frame, "Controls: 'w'=detect keys, 'r'=record, 'q'=quit", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Piano Keys: {len(rectangles)}", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
        # Display the frame
        cv2.imshow("PENANO", frame)
        
        if key == ord('q'):
            break

finally:
    # Cleanup
    if recorder.is_recording():
        recorder.stop_recording()
        filename = recorder.save_to_mp3()
        if filename:
            print(f"Final recording saved as: {filename}")
    
    mixer.cleanup()
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print("Application closed successfully!")