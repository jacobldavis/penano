from ultralytics import YOLO
import cv2
import random
import numpy as np

def getColours(cls_num):
    """Generate unique colors for each class ID"""
    random.seed(cls_num)
    return tuple(random.randint(0, 255) for _ in range(3))


model = YOLO("my_model.pt") 

# Open the default webcam (camera ID 0). Change to another ID if multiple cameras are present.
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

# Store the rectangles to draw
rectangles = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
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
                    # Store rectangle information
                    rectangles.append((x1, y1, x2, y2, colour, class_name, conf))
    
    # Draw the stored rectangles on the current frame
    for rect in rectangles:
        x1, y1, x2, y2, colour, class_name, conf = rect
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
        cv2.putText(frame, f"{class_name} {conf:.2f}", 
                   (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
    
    # Display the frame with rectangles
    cv2.imshow("YOLO Live Detection", frame)
    
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()