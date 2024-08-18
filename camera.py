# https://www.kaggle.com/code/engdhay/hand-detection-by-cv2-and-mediapipe

import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
import mediapipe as mp
import math
import threading

from text_detection import read_text

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1600)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 896)
# print(str(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),str(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

# 1774 1854

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(min_detection_confidence=0.3, min_tracking_confidence=0.3)

hand_draw = mp.solutions.drawing_utils

INDEX_TIP = mp_hands.HandLandmark.INDEX_FINGER_TIP
INDEX_DIP = mp_hands.HandLandmark.INDEX_FINGER_DIP
MIDDLE_TIP = mp_hands.HandLandmark.MIDDLE_FINGER_TIP
MIDDLE_DIP = mp_hands.HandLandmark.MIDDLE_FINGER_DIP

def get_dist(c1, c2):
    return math.sqrt((c1.x-c2.x)**2 + (c1.y-c2.y)**2 + (c1.z-c2.z)**2)

last_hold_gesture = 0
start_pos = None
finish_pos = None

while True:
    ret, frame = cap.read()

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks is not None:
        for hand in result.multi_hand_landmarks:
            index_dist = get_dist(hand.landmark[INDEX_TIP], hand.landmark[MIDDLE_TIP])
            if index_dist < 0.2:
                last_hold_gesture += 1
                if last_hold_gesture == 30:
                    start_pos = hand.landmark[INDEX_TIP]
                    print("start", start_pos)
                elif last_hold_gesture > 30:
                    pass
                else:
                    print("touching")
            elif start_pos is not None:
                last_hold_gesture = 0
                finish_pos = hand.landmark[INDEX_TIP]
                cv2.imwrite("capture.jpg", frame)
                thread = threading.Thread(target=read_text, args=(start_pos, finish_pos))
                thread.start()
                # read_text(start_pos, finish_pos)
                print("finish", finish_pos)


                start_pos = None
                finish_pos = None
            hand_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('hand detection', frame)

    if cv2.waitKey(30) & 0xff == ord('x'):
        break

cap.release()
cv2.destroyAllWindows()