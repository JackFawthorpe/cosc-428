
import cv2
import mediapipe as mp

from draw_landmarks import draw_connections, draw_landmarks, get_default_hand_connections_style, get_default_hand_landmarks_style
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

def print_text(position, text, image):
  # Font settings
  font = cv2.FONT_HERSHEY_SIMPLEX  # You can change the font style
  font_scale = 1  # Font scale (size)
  color = (0, 255, 0)  # Font color in BGR (green)
  thickness = 2  # Font thickness

  # Put the number on the image
  cv2.putText(image, text, position, font, font_scale, color, thickness)
  return image

def draw(image, results):
  image.flags.writeable = True
  image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
  if results.multi_hand_landmarks:
    for idx, landmarks in enumerate(results.multi_hand_landmarks):
      image = draw_connections(image, landmarks)
      draw_landmarks(image, landmarks)

  # Flips the image horizontally to orientate the image to match the fingers infront of the player 
  # Note: All text prints must be after this point to be orientated in the correct dir
  image = cv2.flip(image, 1)

  if results.multi_hand_landmarks:
    image_rows, image_cols, _ = image.shape
    for idx, landmarks in enumerate(results.multi_hand_landmarks):
      thumb_mcp = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
      image = print_text(mp_drawing._normalized_to_pixel_coordinates(1 - thumb_mcp.x, thumb_mcp.y, image_cols, image_rows), f"{idx}", image)

  # Set the position for the number (top-left corner)
  image = print_text((50, 50), "Hello World", image)
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
    cv2.imshow('Binary Game', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()