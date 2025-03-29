import cv2
import mediapipe as mp
import numpy as np

# Hand tracking setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)

# Initialize a blank canvas
canvas = np.zeros((720, 1280, 3), dtype=np.uint8)

# Default color and thickness
color = (0, 0, 255)  # Red
thickness = 5
prev_x, prev_y = 0, 0

# Define color dictionary
color_dict = {
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
    "yellow": (0, 255, 255)
}

# Define buttons (x, y, width, height)
buttons = {
    "Red": (10, 10, 80, 50),
    "Green": (100, 10, 80, 50),
    "Blue": (190, 10, 80, 50),
    "Yellow": (280, 10, 80, 50),
    "Eraser": (370, 10, 80, 50),
    "Clear All": (460, 10, 120, 50),  # New Clear All button
}

# Open webcam
cap = cv2.VideoCapture(0)

def draw_buttons(frame):
    """ Draws color selection buttons on the screen """
    for color_name, (x, y, w, h) in buttons.items():
        if color_name == "Clear All":
            btn_color = (50, 50, 50)  # Dark gray for clear button
        else:
            btn_color = (0, 0, 0) if color_name == "Eraser" else color_dict[color_name.lower()]

        cv2.rectangle(frame, (x, y), (x + w, y + h), btn_color, -1)
        cv2.putText(frame, color_name, (x + 5, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Flip for mirror effect
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # Get the frame size dynamically
    frame_h, frame_w, _ = frame.shape

    # Ensure canvas size matches the frame dynamically
    if canvas.shape[:2] != (frame_h, frame_w):
        canvas = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)

    draw_buttons(frame)  # Display buttons

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            index_finger_tip = hand_landmarks.landmark[8]  # Index finger tip
            x, y = int(index_finger_tip.x * frame_w), int(index_finger_tip.y * frame_h)

            # Check if index finger is touching a button
            for color_name, (bx, by, bw, bh) in buttons.items():
                if bx < x < bx + bw and by < y < by + bh:
                    if color_name == "Clear All":
                        canvas = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)  # Reset canvas
                    else:
                        color = (0, 0, 0) if color_name == "Eraser" else color_dict[color_name.lower()]

            if prev_x == 0 and prev_y == 0:  # First point
                prev_x, prev_y = x, y

            # Draw on the canvas
            if y > 60:  # Prevent drawing over buttons
                cv2.line(canvas, (prev_x, prev_y), (x, y), color, thickness)
                prev_x, prev_y = x, y

    # 🔥 FIXED: Resize canvas dynamically to match frame before merging
    frame = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)

    cv2.imshow("Gesture Drawing App", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
