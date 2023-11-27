import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from keras.models import load_model
from util.GestureUtil import GestureFacade as gf
# initialize mediapipe

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=2, min_detection_confidence=0.6,
                      min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Load the gesture recognizer model
model = load_model('../mp_hand_gesture')

# Load class names
f = open('../Util/gesture.names', 'r')
classNames = f.read().split('\n')
f.close()
cap = cv2.VideoCapture(0)
print(tf.config.list_physical_devices('GPU'))
#gf = GestureFacade()
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
                coordinate = gf.create_coordinate(lm.x, lm.y, lm.z, x, y, c)
                ler.append(coordinate)
                if i < 21:
                    continue
                else:
                    hands_f.append(gf.create_hand_model(ler, result.multi_handedness[j]))
                    i = 0
                    j += 1
            # Drawing landmarksR on frames
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS,
                                  mp_drawing_styles.get_default_hand_landmarks_style(),
                                  mp_drawing_styles.get_default_hand_connections_style())

        # Predict gestures
        for hand in hands_f:
            prediction = model.predict([gf.get_hand_model_parameter(hand)])
            if np.max(prediction) < 0.8:
                continue
            classID = np.argmax(prediction)
            className = classNames[classID]
            g = gf.create_gesture(hand, classID, className)
            gestures.append(g)
    # show the predictions on the frame
        for g in gestures:
            cv2.putText(frame, gf.get_gesture_name(g), gf.get_gesture_position(g),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Show the final output
    cv2.imshow("Output", frame)

    if (cv2.waitKey(1) == ord('q')
            or cv2.getWindowProperty("Output", cv2.WND_PROP_VISIBLE) < 1):
        break

cap.release()
cv2.destroyAllWindows()
