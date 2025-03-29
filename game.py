import cv2
import mediapipe as mp
import pygame
import random
import time

# Initialize pygame
pygame.init()

# Game settings
WIDTH, HEIGHT = 800, 600
BALL_SPEED_X, BALL_SPEED_Y = 7, 7
PADDLE_SPEED = 10
WINNING_SCORE = 5  # First to 5 wins

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)

# Initialize screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Hand Gesture Magic Pong")

# Ball and paddle positions
ball = pygame.Rect(WIDTH//2 - 15, HEIGHT//2 - 15, 30, 30)
player_paddle = pygame.Rect(50, HEIGHT//2 - 70, 20, 140)
ai_paddle = pygame.Rect(WIDTH - 70, HEIGHT//2 - 70, 20, 140)

# MediaPipe Hand Tracking setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Camera setup
cap = cv2.VideoCapture(0)

# Score
player_score = 0
ai_score = 0
font = pygame.font.Font(None, 50)

# Power-up state
power_up_active = False
power_up_start_time = 0

# Game loop
running = True
ball_speed_x, ball_speed_y = BALL_SPEED_X, BALL_SPEED_Y

while running:
    screen.fill(BLACK)
    
    # Capture frame from camera
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Mirror effect
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    frame_h, frame_w, _ = frame.shape

    # Draw paddles and ball
    pygame.draw.rect(screen, BLUE, player_paddle)
    pygame.draw.rect(screen, GREEN, ai_paddle)
    pygame.draw.ellipse(screen, RED, ball)
    pygame.draw.aaline(screen, WHITE, (WIDTH//2, 0), (WIDTH//2, HEIGHT))

    # Detect hand
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            index_finger_tip = hand_landmarks.landmark[8]  # Index finger tip
            y_position = int(index_finger_tip.y * HEIGHT)  # Map to screen height

            # Move paddle
            player_paddle.y = y_position - player_paddle.height // 2

            # Prevent paddle from moving off-screen
            if player_paddle.top < 0:
                player_paddle.top = 0
            if player_paddle.bottom > HEIGHT:
                player_paddle.bottom = HEIGHT

            # Detect closed fist (power-up activation)
            thumb_tip = hand_landmarks.landmark[4]
            pinky_tip = hand_landmarks.landmark[20]

            if abs(thumb_tip.x - pinky_tip.x) < 0.05:  # If hand is closed
                power_up_active = True
                power_up_start_time = time.time()
                player_paddle.height = 200  # Make paddle bigger

    # Ball movement
    ball.x += ball_speed_x
    ball.y += ball_speed_y

    # Ball collision with walls
    if ball.top <= 0 or ball.bottom >= HEIGHT:
        ball_speed_y *= -1  # Reverse direction

    # Ball collision with paddles
    if ball.colliderect(player_paddle) or ball.colliderect(ai_paddle):
        ball_speed_x *= -1  # Reverse direction

    # AI Paddle movement (follows ball)
    if ai_paddle.centery < ball.centery:
        ai_paddle.y += PADDLE_SPEED
    elif ai_paddle.centery > ball.centery:
        ai_paddle.y -= PADDLE_SPEED

    # Ball goes out of bounds
    if ball.left <= 0:  # AI scores
        ai_score += 1
        ball.x, ball.y = WIDTH//2 - 15, HEIGHT//2 - 15
        ball_speed_x *= -1  # Reverse direction
        player_paddle.height = random.choice([80, 100, 120, 140])  # Change paddle size randomly

    if ball.right >= WIDTH:  # Player scores
        player_score += 1
        ball.x, ball.y = WIDTH//2 - 15, HEIGHT//2 - 15
        ball_speed_x *= -1  # Reverse direction
        player_paddle.height = random.choice([80, 100, 120, 140])  # Change paddle size randomly

    # Reset power-up after 5 seconds
    if power_up_active and time.time() - power_up_start_time > 5:
        power_up_active = False
        player_paddle.height = 140  # Reset paddle size

    # Draw scores
    player_text = font.render(f"Player: {player_score}", True, WHITE)
    ai_text = font.render(f"AI: {ai_score}", True, WHITE)
    screen.blit(player_text, (20, 20))
    screen.blit(ai_text, (WIDTH - 150, 20))

    # Check if someone won
    if player_score == WINNING_SCORE or ai_score == WINNING_SCORE:
        screen.fill(BLACK)
        winner_text = font.render("YOU WIN!" if player_score == WINNING_SCORE else "AI WINS!", True, WHITE)
        screen.blit(winner_text, (WIDTH//2 - 100, HEIGHT//2))
        pygame.display.flip()
        pygame.time.delay(3000)  # Show the result for 3 seconds
        running = False  # Exit the game loop

    # Show frame
    pygame.display.flip()

    # Check for quit event
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

# Cleanup
cap.release()
cv2.destroyAllWindows()
pygame.quit()
