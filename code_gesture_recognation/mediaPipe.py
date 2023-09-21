import cv2
import mediapipe as mp

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    x, y, c = frame.shape
    # Flip the frame vertically
    frame = cv2.flip(frame, 1)
    #bgr(cv2) to rgb(mp)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Get hand landmark prediction
    result = hands.process(framergb)
    # post process the result
    if result.multi_hand_landmarks:

        for handslms in result.multi_hand_landmarks:
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

    # Show the final output
    cv2.imshow("Output", frame)

    if (cv2.waitKey(1) == ord('q')
            or cv2.getWindowProperty("Output", cv2.WND_PROP_VISIBLE) < 1):
        break

# release the webcam and destroy all active windows
cap.release()
cv2.destroyAllWindows()