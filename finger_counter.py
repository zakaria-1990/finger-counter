import cv2
import mediapipe as mp
import numpy
import os
import hand_tracking_module as htm

cap = cv2.VideoCapture(0)

detector = htm.HandDetector(detection_con=0.75)

tips_ids = [8, 12, 16, 20]

while cap.isOpened():
    success, img = cap.read()
    img = detector.find_hands(img)
    landmark_list = detector.find_position(img, draw=False)
    # print(landmark_list)

    if landmark_list != None:
        fingers = []
        # thumb case
        if landmark_list[4][1] > landmark_list[3][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        # other fingers
        for finger in tips_ids:
            if landmark_list[finger][2] < landmark_list[finger - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        total_fingers = fingers.count(1)

        cv2.putText(img, str(total_fingers), (20, 100), cv2.FONT_HERSHEY_PLAIN,
                    5, (255, 0, 0), 10)


    cv2.imshow('finger counter', img)
    if cv2.waitKey(5) & 0XFF == 27:
        break

cap.release()
cv2.destroyAllWindows()