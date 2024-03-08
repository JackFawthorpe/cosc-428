
import cv2
import mediapipe as mp

from draw_landmarks import draw_connections, draw_landmarks, get_default_hand_connections_style, get_default_hand_landmarks_style
mp_hands = mp.solutions.hands
cap = cv2.VideoCapture(0)

def draw(image, results):
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      for landmarks in results.multi_hand_landmarks:
        image = draw_connections(image, landmarks)
        draw_landmarks(image, landmarks)
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
    image = draw(image, results)

    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('Binary Game', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()