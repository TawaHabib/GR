import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from keras.models import load_model
from util.GestureUtil import Gesture
from util.GestureUtil import Hand
from util.GestureUtil import PixelCoordinate
from util.GestureUtil import HandProperty
# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=2, min_detection_confidence=0.8)
mpDraw = mp.solutions.drawing_utils

# Load the gesture recognizer model
model = load_model('../mp_hand_gesture')

# Load class names
f = open('../Util/gesture.names', 'r')
classNames = f.read().split('\n')
f.close()
cap = cv2.VideoCapture(0)


while True:
    # Read each frame from the webcam
    _, frame = cap.read()
    x, y, c = frame.shape
    # Flip the frame vertically
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get hand landmark prediction
    result = hands.process(framergb)

    className = ''
    landmarks1 = []
    hands_f = []
    gestures = []

    # post process the result
    if result.multi_hand_landmarks:
        i = 0
        j = 0
        for handslms in result.multi_hand_landmarks:
            ler = []
            for lm in handslms.landmark:
                i += 1
                coordinate = PixelCoordinate(lm.x, lm.y, lm.z, x, y, c)
                ler.append(coordinate)
                if i < 21:
                    continue
                else:
                    hands_f.append(Hand(ler, result.multi_handedness[j]))
                    i = 0
                    j += 1

            # Drawing landmarksR on frames
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

            # Predict gestures

            for hand in hands_f:
                prediction = model.predict([hand.get_hand_lm_list()])
                classID = np.argmax(prediction)
                className = classNames[classID]
                g = Gesture(hand, className, classID)
                gestures.append(g)

    # show the predictions on the frame
    for g in gestures:
        print(g.hand.property_hand.label)
        cv2.putText(frame, g.gesture_name, [int(g.hand.sum_x/21), int(g.hand.sum_y/21)], cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2, cv2.LINE_AA)

    # Show the final output
    cv2.imshow("Output", frame)

    if (cv2.waitKey(1) == ord('q')
            or cv2.getWindowProperty("Output", cv2.WND_PROP_VISIBLE) < 1):
        break

cap.release()
cv2.destroyAllWindows()
