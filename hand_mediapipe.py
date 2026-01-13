import cv2                  # Camera handling
import mediapipe as mp      # Hand tracking
import pyautogui            # Mouse control
import math                 # Distance calculation

cap = cv2.VideoCapture(0)   # Open webcam

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)   # Detect only one hand
mp_draw = mp.solutions.drawing_utils

screen_w, screen_h = pyautogui.size()     # Screen resolution

prev_x, prev_y = 0, 0                     # Previous mouse position
smoothening = 7                           # Smoothness factor

while True:
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)            # Mirror camera

    h, w, _ = frame.shape

    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)       # Detect hand

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

            # Index finger tip (ID = 8)
            index_tip = hand_landmarks.landmark[8]
            ix, iy = int(index_tip.x * w), int(index_tip.y * h)

            # Thumb tip (ID = 4)
            thumb_tip = hand_landmarks.landmark[4]
            tx, ty = int(thumb_tip.x * w), int(thumb_tip.y * h)

            # Red dot on thumb
            cv2.circle(frame, (tx, ty), 8, (0, 0, 255), cv2.FILLED)

            # Green dot on index finger
            cv2.circle(frame, (ix, iy), 8, (0, 255, 0), cv2.FILLED)

            # Mouse movement (index finger)
            screen_x = int(index_tip.x * screen_w)
            screen_y = int(index_tip.y * screen_h)

            curr_x = prev_x + (screen_x - prev_x) / smoothening
            curr_y = prev_y + (screen_y - prev_y) / smoothening
            pyautogui.moveTo(curr_x, curr_y)
            prev_x, prev_y = curr_x, curr_y

            # Distance for click
            distance = math.hypot(ix - tx, iy - ty)

            if distance < 40:
                pyautogui.click()
                cv2.putText(frame, "CLICK", (40, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 0), 2)

    cv2.imshow("Virtual Mouse (Simple Marks)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
