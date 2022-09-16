import cv2
import mediapipe as mp
from math import floor
from playsound import playsound

cap = cv2.VideoCapture(0)
mphands = mp.solutions.hands
hands = mphands.Hands()
mp_draw = mp.solutions.drawing_utils

notes = {0 : "Do", 1 : "Re", 2 : "Mi", 3 : "Fa", 4 : "So", 5 : "La", 6 : "Si"}

while True:
    success, img = cap.read()
    
    imgw = int(img.shape[1])
    imgh = int(img.shape[0])
    
    cv2.line(img, (0, int(0.25 * imgh)), (imgw, int(0.25 * imgh)), (255, 0, 0), 2)
    for i in range(0, 150 * 9, 150):
        cv2.line(img, (i, 0), (i, int(0.25 * imgh)), (255, 0, 0), 2)
        if i / 150 < 7:
            cv2.putText(img, notes[i / 150], (i + 20, int(0.15 * imgh)),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 3)
    
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # mp works with RGB image only
    results = hands.process(imgRGB) # recognizing hand
    
    if results.multi_hand_landmarks:
        for hand_loc in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_loc, mphands.HAND_CONNECTIONS)
            
            tip = hand_loc.landmark[mphands.HandLandmark.INDEX_FINGER_TIP]
            if 0.2 < tip.y < 0.25 and tip.x * imgw / 150 < 7:
                playsound(notes[floor(tip.x * imgw / 150)] + ".wav")
                
    cv2.imshow("Image", cv2.flip(img, 1))
    cv2.waitKey(1)