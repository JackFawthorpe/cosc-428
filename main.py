
import math
import cv2
import mediapipe as mp

from draw_hand import draw_connections, draw_hand, draw_landmarks, get_default_hand_connections_style, get_default_hand_landmarks_style
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)


# Index's of the tips of fingers against their bases
BINARY_ON_CHECKS = [(mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_MCP),
                     (mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_MCP),
                       (mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_MCP),
                         (mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_MCP)]


def print_text(position, text, image):
  # Font settings
  font = cv2.FONT_HERSHEY_SIMPLEX  # You can change the font style
  font_scale = 1  # Font scale (size)
  color = (0, 255, 0)  # Font color in BGR (green)
  thickness = 2  # Font thickness

  # Put the number on the image
  cv2.putText(image, text, position, font, font_scale, color, thickness)
  return image

def annotate_image(image, binary_result):
  if hand_processing.multi_hand_landmarks:
    image_rows, image_cols, _ = image.shape
    for idx, landmarks in enumerate(hand_processing.multi_hand_landmarks):
      thumb_mcp = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
      image = print_text(mp_drawing._normalized_to_pixel_coordinates(1 - thumb_mcp.x, thumb_mcp.y, image_cols, image_rows), f"{idx}", image)

  # Set the position for the number (top-left corner)
  image = print_text((50, 50), f"{binary_result}", image)
  return image

FINGER_DISTANCE_ERROR = 0.1

def get_binary(hand_processing):
  if not hand_processing.multi_hand_landmarks:
    return 0
  output = 0
  for hand_index, landmarks in enumerate(hand_processing.multi_hand_landmarks):
    for power_index in range(4):
      tip =  landmarks.landmark[BINARY_ON_CHECKS[power_index][0]]
      dip =  landmarks.landmark[BINARY_ON_CHECKS[power_index][1]]
      if abs(tip.y - dip.y) > FINGER_DISTANCE_ERROR:
        output += 2 ** power_index
  return output

with mp_hands.Hands(
    model_complexity=0,
    max_num_hands=2,
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
    hand_processing = hands.process(image)
    binary_result = get_binary(hand_processing)

    # Draw the hand annotations on the image.
    image = draw_hand(image, hand_processing)
    
    # Flips the image horizontally to orientate the image to match the fingers infront of the player 
    # Note: All text prints must be after this point to be orientated in the correct dir
    image = cv2.flip(image, 1)

    image = annotate_image(image, binary_result)
    cv2.imshow('Binary Game', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()