
import cv2
import mediapipe as mp

from draw_landmarks import get_default_hand_connections_style, get_default_hand_landmarks_style

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
cap = cv2.VideoCapture(0)

# Removes the thumb from the connections to draw
HAND_CONNECTIONS = frozenset({(17, 18), (0, 17), (13, 14), (13, 17), (18, 19), (5, 6), (5, 9), (14, 15), (0, 5), (9, 10), (9, 13), (10, 11), (19, 20), (6, 7), (15, 16), (11, 12), (7, 8)})

def draw_landmarks(image, results):
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            HAND_CONNECTIONS,
            get_default_hand_landmarks_style(),
            get_default_hand_connections_style())
    return image

with mp_hands.Hands(
    model_complexity=0,
    max_num_hands=4,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()

    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)


    # Draw the hand annotations on the image.
    image = draw_landmarks(image, results)

    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('Binary Game', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()