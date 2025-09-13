# To run this script, you need to install opencv-python and mediapipe
# pip install opencv-python mediapipe

import cv2
import mediapipe as mp
import time
import numpy as np

# --- Constants and Setup ---
# MediaPipe Hands setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Landmark indices for fingertips
FINGER_TIPS = [4, 8, 12, 16, 20]

# --- Jolt Detection Thresholds ---
# How high the Z-acceleration must be to register a "jolt".
# This value is sensitive and may need tuning.
JOLT_THRESHOLD = 0.018

# Cooldown period in seconds to prevent a single jolt from being
# registered multiple times.
JOLT_COOLDOWN = 0.4

# --- State Variables ---
# Stores the toggle state for each finger (False=released, True=clicked)
finger_clicked_state = [False] * 5

# Stores motion data for each finger to calculate acceleration
# [current_pos, previous_pos, previous_velocity]
finger_motion_data = [[0, 0, 0] for _ in range(5)]
last_jolt_time = [0] * 5

# --- Main Program ---
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Jolt detection active. Press 'q' to quit.")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Flip the image horizontally and convert BGR to RGB
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    
    # To improve performance, mark the image as not writeable
    image.flags.writeable = False
    results = hands.process(image)

    # Revert image for display
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # --- Detection Phase ---
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        
        current_time = time.time()

        for i in range(5):
            # 1. Get current position and update motion data history
            tip_z = hand_landmarks.landmark[FINGER_TIPS[i]].z
            
            prev_pos_z = finger_motion_data[i][1]
            prev_velocity_z = finger_motion_data[i][2]
            
            finger_motion_data[i][0] = tip_z      # Current position
            finger_motion_data[i][1] = prev_pos_z # Old position becomes previous

            # 2. Calculate velocity and acceleration
            # Velocity = change in position
            velocity = tip_z - prev_pos_z
            # Acceleration = change in velocity
            acceleration = velocity - prev_velocity_z
            
            # Update previous velocity for the next frame
            finger_motion_data[i][2] = velocity

            # 3. Check for a jolt (high acceleration) after cooldown
            if current_time - last_jolt_time[i] > JOLT_COOLDOWN:
                if abs(acceleration) > JOLT_THRESHOLD:
                    # A jolt was detected, toggle the state
                    finger_clicked_state[i] = not finger_clicked_state[i]
                    
                    action = "CLICKED" if finger_clicked_state[i] else "UN-CLICKED"
                    print(f"Jolt on finger {i}: {action}")

                    # Start the cooldown
                    last_jolt_time[i] = current_time

            # 4. Visual Feedback
            if finger_clicked_state[i]:
                tip_landmark = hand_landmarks.landmark[FINGER_TIPS[i]]
                tip_x = int(tip_landmark.x * image.shape[1])
                tip_y = int(tip_landmark.y * image.shape[0])
                cv2.circle(image, (tip_x, tip_y), 10, (0, 0, 255), -1)

        mp_drawing.draw_landmarks(
            image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
    # --- Display Info ---
    # Convert boolean list to int list for cleaner display
    display_state = [int(s) for s in finger_clicked_state]
    state_text = f"Click State: {display_state}"
    cv2.putText(image, state_text, (50, image.shape[0] - 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Display the final image
    cv2.imshow('Finger Jolt Detection', image)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()
hands.close()
print("Script finished.")
