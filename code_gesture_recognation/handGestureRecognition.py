import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from keras.models import load_model
from util.GestureUtil import GestureUtil

# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=2, min_detection_confidence=0.8)
mpDraw = mp.solutions.drawing_utils

# Load the gesture recognizer model
model = load_model('../mp_hand_gesture')
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

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
    gestures = []

    # post process the result
    if result.multi_hand_landmarks:
        i = 0
        for handslms in result.multi_hand_landmarks:
            ler = []
            #print(handslms.landmark)
            for lm in handslms.landmark:
                i += 1
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)
                ler.append([lmx, lmy])
                if i < 21:
                    continue
                else:
                    landmarks1.append([ler])
                    i = 0
            # Drawing landmarksR on frames
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

            # Predict gestures
            for landm in landmarks1:
                prediction = model.predict(landm)
                classID = np.argmax(prediction)
                className = classNames[classID]
                g = GestureUtil(landm, className, x, y)
                gestures.append(g)

    # show the predictions on the frame
    for g in gestures:
        cv2.putText(frame, g.gestureName, g.puntoMano, cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2, cv2.LINE_AA)

    # Show the final output
    cv2.imshow("Output", frame)

    if (cv2.waitKey(1) == ord('q')
            or cv2.getWindowProperty("Output", cv2.WND_PROP_VISIBLE) < 1):
        break

cap.release()
cv2.destroyAllWindows()
