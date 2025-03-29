import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Start video capture
cap = cv2.VideoCapture(0)

# Finger landmark indexes
finger_tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky
finger_dips = [6, 10, 14, 18]  # Corresponding DIP joints

# Variables for waving detection
previous_x = None
wave_counter = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame for mirror effect
    frame = cv2.flip(frame, 1)

    # Convert BGR to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    gesture_text = ""

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Count raised fingers (Index, Middle, Ring, Pinky)
            raised_fingers = [
                tip for tip, dip in zip(finger_tips, finger_dips)
                if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[dip].y
            ]

            # Thumb check
            thumb_tip = hand_landmarks.landmark[4]
            thumb_ip = hand_landmarks.landmark[2]
            thumb_extended = thumb_tip.x < thumb_ip.x  # For right hand (Flipped video)

            # Distance between index finger and thumb (for "Pookie")
            index_tip = hand_landmarks.landmark[8]
            distance = np.sqrt((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2)

            # Get wrist position for waving detection
            wrist_x = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x

            # Check if hand is moving left-right quickly
            if previous_x is not None:
                if abs(wrist_x - previous_x) > 0.05:  # Movement threshold
                    wave_counter += 1
                else:
                    wave_counter = max(0, wave_counter - 1)

            previous_x = wrist_x

            # Gesture Recognition
            if thumb_extended and len(raised_fingers) == 0:
                gesture_text = "Thumbs Up "
            elif set(raised_fingers) == {8, 12}:  # Index + Middle
                gesture_text = "Victory "
            elif raised_fingers == [12]:  # Only Middle Finger
                gesture_text = "F*** Off "
            elif distance < 0.05:  # Thumb and Index Finger Touching
                gesture_text = "Pookie"
            elif len(raised_fingers) == 4 and wave_counter > 6:  # Waving motion detected
                gesture_text = "Byee "

    # Display gesture text
    if gesture_text:
        cv2.putText(frame, gesture_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("Hand Gesture Recognition", frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
