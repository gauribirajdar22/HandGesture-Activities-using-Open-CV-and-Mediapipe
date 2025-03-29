import cv2
import mediapipe as mp
import os
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)

# Define the folder where images are stored
image_folder = r"C:\Users\biraj\OneDrive\Documents\HandGestureProject\Imagess"

# Get list of images in the folder
image_files = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(('.jpg', '.jpeg', '.png'))]
current_index = 0

# Load the first image to avoid undefined variable issue
image = cv2.imread(image_files[current_index])
image = cv2.resize(image, (640, 480))

# Variables for gesture tracking
last_x = None
swipe_threshold = 50  # Minimum movement required for swipe
zoom_factor = 1.0

def detect_swipe(hand_landmarks, width):
    """Detects left or right swipe motion based on wrist movement."""
    global last_x, current_index

    wrist_x = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * width)

    if last_x is None:
        last_x = wrist_x
        return current_index

    movement = wrist_x - last_x  # Positive → Right, Negative → Left

    if abs(movement) > swipe_threshold:
        if movement > 0:
            current_index = (current_index + 1) % len(image_files)  # Swipe Right
        else:
            current_index = (current_index - 1) % len(image_files)  # Swipe Left
        last_x = wrist_x  # Update position after swipe

    return current_index

def detect_zoom(hand_landmarks, width, height):
    """Adjusts zoom factor based on distance between thumb and index finger."""
    global zoom_factor

    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

    thumb_x, thumb_y = int(thumb_tip.x * width), int(thumb_tip.y * height)
    index_x, index_y = int(index_tip.x * width), int(index_tip.y * height)

    distance = ((thumb_x - index_x) ** 2 + (thumb_y - index_y) ** 2) ** 0.5

    # If fingers move apart, zoom in. If they come closer, zoom out.
    if distance > 100:  # Fingers spread → Zoom In
        zoom_factor = min(2.5, zoom_factor + 0.05)
    elif distance < 50:  # Fingers close → Zoom Out
        zoom_factor = max(1.0, zoom_factor - 0.05)

def apply_zoom(image, zoom_factor):
    """Applies zoom effect by cropping and resizing."""
    h, w, _ = image.shape
    zoom_w = int(w / zoom_factor)
    zoom_h = int(h / zoom_factor)

    start_x = (w - zoom_w) // 2
    start_y = (h - zoom_h) // 2
    end_x = start_x + zoom_w
    end_y = start_y + zoom_h

    cropped = image[start_y:end_y, start_x:end_x]
    return cv2.resize(cropped, (640, 480))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape
    frame = cv2.flip(frame, 1)  # Flip for mirror effect
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                current_index = detect_swipe(hand_landmarks, width)  # Swipe left/right
                detect_zoom(hand_landmarks, width, height)  # Pinch-to-zoom

    # Load and apply zoom to the updated image (this happens outside the hand detection block)
    image = cv2.imread(image_files[current_index])
    image = cv2.resize(image, (640, 480))
    image = apply_zoom(image, zoom_factor)

    cv2.imshow("Hand Gesture Image Viewer", image)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
