import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Start video capture
cap = cv2.VideoCapture(0)

# Finger landmark indexes
finger_tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky
finger_dips = [6, 10, 14, 18]  # Corresponding DIP joints

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame for mirror effect
    frame = cv2.flip(frame, 1)

    # Convert BGR to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    # Default finger count
    finger_count = 0  

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Count fingers (Index, Middle, Ring, Pinky)
            finger_count = sum(
                1 for tip, dip in zip(finger_tips, finger_dips)
                if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[dip].y
            )

            # Check if the thumb is extended
            thumb_tip = hand_landmarks.landmark[4]
            thumb_ip = hand_landmarks.landmark[2]

            if thumb_tip.x < thumb_ip.x:  # Thumb is extended if it's on the left (right-hand case)
                finger_count += 1

    # Determine Even/Odd status
    status = "EVEN" if finger_count % 2 == 0 else "ODD"

    # Display results
    cv2.putText(frame, f"Fingers: {finger_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Status: {status}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Show the frame
    cv2.imshow("Hand Gesture Recognition", frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
